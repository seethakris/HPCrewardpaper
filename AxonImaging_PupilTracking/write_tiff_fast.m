function [ ] = write_tiff_fast( outfile, data, varargin )
%FASTWRITE_TIF Write TIFF file quickly
% Use file I/O to write tiff data more efficiently than the built-in Matlab
% protocols. A lot of efficiency is gained when we assume that the non-data
% values for each image are the same, and it is assumed to be so here.
%
% INPUT:
%   outfile - The output file location (will overwrite)
%   data - Either a 3D matrix containing the stack of data or a function
%       that takes two integer arguments, the start index and the end
%       index, that returns a 3D matrix of the data from those indicies.
%
% OPTIONAL PARAMETERS:
%   tiff_fields ([]): The properties of the TIFF data. Can either be blank,
%       and then the fields will be guessed from the data, or can be a
%       reference to a TIFF file.
%   header_length (104): The length of the header. Must be at least 8
%   start_ind (1): The first index to write
%   end_ind (inf): The highest index to write. If inf and data is a matrix,
%       it will use the size of data, otherwise end_ind must be provided
%   nsplits (1): The number of files to separate the interleaved data
%       across.
%
% Updated 11-10-2016 by Dr. Jason R. Climer (jason.r.climer@gmail.com)

% Parse inputs
% keyboard
mode = [];

ip=inputParser();
ip.addParameter('tiff_fields',[]);
ip.addParameter('header_length',8);
ip.addParameter('start_ind',1);
ip.addParameter('end_ind',inf);
ip.addParameter('max_write_buff',inf);
ip.addParameter('calcmult',6);
ip.addParameter('mode','imageJ_raw');
ip.addParameter('endian','l');

ip.parse(varargin{:});
for j=fields(ip.Results)'
    eval([j{1} '=ip.Results.' j{1} ';']);
end

tiff_sizes_tbl = [1 1 2 4 8]';% Byte counts for each "size" value in the IFD tag

if ~isequal(class(data),'function_handle')% If data isn't a function handle
    if isinf(end_ind), end_ind = size(data,3); end% Set end_ind if necessary
    data = @(i,j)data(:,:,i:j);% Make a function handle for the matrix
else
    if isinf(end_ind)% Assert that end_ind is provided
        ME = MException('fastwrite_tif:noEndInd', ...
            'You must provide an end ind for functional data.');
        throw(ME)
    end
end

vect = @(x)x(:);% Helper function
Tags = fields(Tiff.TagID);% Possible tags

%%
%%%%%%%%%%%% Make tiff_fields struct %%%%%%%%%%%%%%%%%%
if isempty(tiff_fields)% Guess based on data & defaults
    %     keyboard
    tiff_fields = struct();
    tiff_fields.ImageLength = struct('sze',3,'n',1,'value',size(data(1,1),1));
    tiff_fields.ImageWidth = struct('sze',3,'n',1,'value',size(data(1,1),2));
    tiff_fields.Photometric = struct('sze',3,'n',1,'value',1);
    tiff_fields.BitsPerSample = struct('sze',3,'n',1,'value',max(8*ceil(nextpow2(max(vect(data(1,1))))/8),16));
    tiff_fields.SamplesPerPixel = struct('sze',3,'n',1,'value',1);
    
    if mod(tiff_fields.ImageLength.value,16)==0
        tiff_fields.RowsPerStrip = struct('sze',3,'n',1,'value',16);
    else
        tiff_fields.RowsPerStrip =  struct('sze',3,'n',1,'value',tiff_fields.ImageLength.value);
    end
    tiff_fields.PlanarConfiguration = struct('sze',3,'n',1,'value',1);
    
    for i=fields(tiff_fields)'
        tiff_fields.(i{1}).tag = Tiff.TagID.(i{1});
    end
elseif ischar(tiff_fields)% Copy fields from the file given
    T = fopen(tiff_fields);
    
    switch fread(T,1,'uint16')
        case 18761            
            fclose(T);
            T = fopen(tiff_fields,'r','l');
        case 19789
            fclose(T);
            T = fopen(tiff_fields,'r','b');
    end
    
    
    tiff_fields = struct();
    
    fseek(T,4,-1);% Move to the offset of the offset of the first IFD
    fseek(T,fread(T,1,'uint32'),-1);% Move to the offset of the first IFD
    
    for i=1:fread(T,1,'uint16')% Each tag
        tag = fread(T,1,'uint16');% Get the tag
        tiff_fields.(Tags{cellfun(@(x)Tiff.TagID.(x),Tags)==tag}).tag = tag;% Store the tag
        tag = Tags{cellfun(@(x)Tiff.TagID.(x),Tags)==tag};% Get the tag name
        
        tiff_fields.(tag).sze = fread(T,1,'uint16');% Write the size
        tiff_fields.(tag).n = fread(T,1,'uint32');% Write the length
        if tiff_fields.(tag).sze==3&&tiff_fields.(tag).n==1
            tiff_fields.(tag).value = fread(T,1,'uint16');% Write the value
            fread(T,1,'uint16');
        else
        tiff_fields.(tag).value = fread(T,1,'uint32');% Write the value
        end
        last_loc = ftell(T);
        if tiff_sizes_tbl(tiff_fields.(tag).sze)*tiff_fields.(tag).n>4% If the value is binary data
            fseek(T,tiff_fields.(tag).value,-1);% Go to the position of the data
            switch tiff_fields.(tag).sze% How to read the data
                case 1% Bytes
                    tiff_fields.(tag).value = fread(T,tiff_fields.(tag).n,'uint8');
                case 2% ASCII
                    tiff_fields.(tag).value = char(fread(T,tiff_fields.(tag).n,'uint8'));
                case 3% uint16 (short)
                    tiff_fields.(tag).value = fread(T,tiff_fields.(tag).n,'uint16');
                case 4% uint32 (long)
                    tiff_fields.(tag).value = fread(T,tiff_fields.(tag).n,'uint32');
                case 5% two longs (rational)
                    tiff_fields.(tag).value = reshape(fread(T,tiff_fields.(tag).n*2,'uint32'),[2 tiff_fields.(tag).n]);
            end
        end
        fseek(T,last_loc,-1);
    end
end

% Calculate the number of strips in each image
StripsPerImage = tiff_fields.ImageLength.value/tiff_fields.RowsPerStrip.value;

tags = fields(tiff_fields);% The tags to write
if ~ismember(tags,'StripOffsets')
    tiff_fields.StripOffsets = struct('tag',Tiff.TagID.('StripOffsets'),'sze',4,'n',StripsPerImage,'value',[]);
    tags = [tags; {'StripOffsets'}];
end
if ~ismember(tags,'StripByteCounts')
%     keyboard
    temp = repmat(tiff_fields.RowsPerStrip.value,ceil(StripsPerImage),1);
    temp(end) = temp(end)-(tiff_fields.ImageLength.value-sum(temp));
    tiff_fields.StripByteCounts = struct('tag',Tiff.TagID.('StripOffsets'),'sze',4,'n',ceil(StripsPerImage),'value',temp*tiff_fields.ImageWidth.value*double(tiff_fields.BitsPerSample.value)/8);
%     if StripsPerImage-floor(StripsPerImage)>0
%         tiff_fields.StripByteCounts.value=[tiff_fields.StripByteCounts.value;...
%             (tiff_fields.ImageLength.value-tiff_fields.RowsPerStrip.value*floor(StripsPerImage))*tiff_fields.ImageWidth.value*tiff_fields.BitsPerSample.value/8];
%     end
    tags = [tags; {'StripByteCounts'}];
end

if isequal(mode,'auto')
    if (end_ind-start_ind+1)*tiff_fields.ImageLength.value*tiff_fields.ImageWidth.value*tiff_fields.BitsPerSample.value/8>2147483648
        mode = 'imageJ_raw';
    else
        mode = 'tiff42';
    end
end

% Find the shared values and update their locations
if isequal(mode,'imageJ_raw')
    % Remake tiff_fields
    tiff_fields_old = tiff_fields;
    tiff_fields = struct;
    ImageDescription = sprintf([...
        'ImageJ=1.49k\n'...
        'images=%i\n'...
        'slices=%i\n'...
        'loop=false\n'...
        'min=00000.0\n'...
        'max=00000.0'...
        ],(end_ind-start_ind+1),(end_ind-start_ind+1));
    
        tiff_fields.ImageDescription = struct(...
            'tag',Tiff.TagID.('ImageDescription')...
            ,'n',numel(ImageDescription)...
            ,'sze',2 ...
            ,'value',ImageDescription...
            );
        
     StripsPerImage = 1;
    
    for i={'StripOffsets','ImageWidth','ImageLength','BitsPerSample','Photometric','SamplesPerPixel','RowsPerStrip'}
        tiff_fields.(i{1}) = tiff_fields_old.(i{1});
    end
    tiff_fields.RowsPerStrip.value = tiff_fields.ImageLength.value;
    tiff_fields.StripByteCounts = struct(...
        'tag',279 ...
    ,'sze',4 ...
    ,'n',1 ...
    ,'value',tiff_fields.ImageWidth.value*tiff_fields.RowsPerStrip.value*tiff_fields.BitsPerSample.value/8);

    tags = fields(tiff_fields);
end

[~,i] = sort(cellfun(@(x)Tiff.TagID.(x),tags));
tags = tags(i);

shared_inds = find(tiff_sizes_tbl(cellfun(@(tag)tiff_fields.(tag).sze,tags)).*cellfun(@(tag)tiff_fields.(tag).n,tags)>4&...
    ~ismember(tags,'StripOffsets'));
shared_offsets = cumsum([0;cellfun(@(x)tiff_fields.(x).n*tiff_sizes_tbl(tiff_fields.(x).sze),tags(shared_inds))]);

if isequal(mode,'imageJ_raw')
    tiff_fields.StripOffsets.n=1;
    tiff_fields.StripOffsets.value=header_length+2+(numel(tags)+1)*12+4+shared_offsets(end);
end

%%
% keyboard
for j=fields(tiff_fields)'
    if ~isequal(class(tiff_fields.(j{1}).value),'double')
        tiff_fields.(j{1}).value = double(tiff_fields.(j{1}).value);
    end
end

% tiff_fields = repmat(tiff_fields,numel(outfile),1);
% j = cell(numel(outfile),1);
% [j{:}] = data(1,1);

%%
O = fopen(outfile,'w',endian);% Open the output file
if O==-1
    keyboard
end

% Write the header
switch endian
    case {'b','ieee-be'}
        fwrite(O,19789,'uint16');
%         binfun = @(x)[bin2dec(x(17:32)) bin2dec(x(1:16))];
%         binfun = @(x)[bin2dec(x(16:-1:1)) bin2dec(x(32:-1:17))];
    binfun = @(x)[bin2dec(x(1:16)) bin2dec(x(17:32))];
%     binfun = @(x)[bin2dec(x(32:-1:17)) bin2dec(x(16:-1:1))];
    case {'l','ieee-le'}
        fwrite(O,18761,'uint16');
        binfun = @(x)[bin2dec(x(17:32)) bin2dec(x(1:16))];
end
    
fwrite(O,42,'uint16');
fwrite(O,header_length,'uint32');
fwrite(O,zeros(header_length-8,1),'uint8');

% To write all the IFDs at once, we're going to use a buffer



buffer = zeros(1+6*numel(tags)+2,1);
buffer(1) = numel(tags);% The number of tags in each IFD
buffer(2:(2+4):(2+4)*(numel(tags)))=cellfun(@(x)Tiff.TagID.(x),tags);% The TagIDs
buffer((2:(2+4):(2+4)*(numel(tags)))+1)=cellfun(@(x)tiff_fields.(x).sze,tags);% The sizes of each tag

% The length of the tag is stored as a long (uint32). To make writing more
% efficent, the whole buffer is expressed as shorts (uint16)
j=cellfun(binfun...% Get the shorts that equal the same long value
    ,arrayfun(...
    @(x)dec2bin(x,32)...% Convert to binary
    ,cellfun(@(x)tiff_fields.(x).n,tags)...% The length of the tag values
    ,'UniformOutput',false)...
    ,'UniformOutput',false);
j = cat(1,j{:})';% Format in order
buffer(sort([(2:(2+4):(2+4)*(numel(tags)))+2 (2:(2+4):(2+4)*(numel(tags)))+3]))=...
    j(:);

% Values - also longs as above. This is only for values that have a total
% size under 4 bytes. If it's a long, use full long, otherwise use the
% short
k = find(~ismember(1:numel(tags),shared_inds)'&~ismember(tags,'StripOffsets')&~(cellfun(@(x)tiff_fields.(x).sze,tags)==3&cellfun(@(x)tiff_fields.(x).n,tags)==1));
j=cellfun(binfun...% Get the shorts
    ,arrayfun(@(x)dec2bin(x,32),...% Convert to binary
    cellfun(@(x)tiff_fields.(x).value,tags(k))...% The value
    ,'UniformOutput',false)...
    ,'UniformOutput',false);
j = cat(1,j{:})';% Format in order
buffer(sort([6*k;6*k+1]))=j(:);

k = find(~ismember(1:numel(tags),shared_inds)'&~ismember(tags,'StripOffsets')&(cellfun(@(x)tiff_fields.(x).sze,tags)==3&cellfun(@(x)tiff_fields.(x).n,tags)==1));
buffer(6*k) = cellfun(@(x)double(tiff_fields.(x).value),tags(k));

% Shared - These values are the same between all the IFDs and are thus
%   shared in the memory.
j=cellfun(binfun...% Get the shorts
    ,arrayfun(@(x)dec2bin(x,32),...% Convert to binary
    ... The lines below calculate the location of the shared data
    header_length...
    +((end_ind-start_ind+1)*(~isequal(mode,'imageJ_raw'))+isequal(mode,'imageJ_raw'))*(6+12*(numel(tags)+isequal(mode,'imageJ_raw')))... The end of the IFDs
    +shared_offsets(1:end-1)... The offset to the shared data
    ,'UniformOutput',false)...
    ,'UniformOutput',false);
j = cat(1,j{:})';% Format in order
buffer(sort([6*shared_inds;6*shared_inds+1]))=j(:);

if isequal(mode,'tiff42')
    % Set up the buffer
    maxbuff = memory;% Ask available memory
    maxbuff = min(maxbuff.MaxPossibleArrayBytes-maxbuff.MemUsedMATLAB,max_write_buff);
    maxbuff = min(2^(nextpow2(maxbuff/numel(buffer))-1), end_ind-start_ind+1);% How big for the buffer
    buffer = repmat(buffer,[1 maxbuff]);% Repeat the shared data
    k = find(ismember(tags,'StripOffsets'));% Index of the strip offsets
    
    for i=1:ceil((end_ind-start_ind+1)/maxbuff)
        if StripsPerImage>1
        j=cellfun(binfun...% Get the shorts
            ,arrayfun(@(x)dec2bin(x,32),...% Convert to binary
            header_length...
            +(end_ind-start_ind+1)*(6+12*numel(tags))...The end of the IFDs
            +shared_offsets(end)...The end of the shared data
            +((i-1)*maxbuff:min(end_ind-start_ind,i*maxbuff-1))*4*StripsPerImage...% The offset of the start of each of the stripoffsets
            ,'UniformOutput',false)...
            ,'UniformOutput',false);        
        else
            j=cellfun(binfun...% Get the shorts
            ,arrayfun(@(x)dec2bin(x,32),...% Convert to binary
            header_length...
            +(end_ind-start_ind+1)*(6+12*numel(tags))...The end of the IFDs
            +shared_offsets(end)...The end of the shared data
            +((i-1)*maxbuff:min(end_ind-start_ind,i*maxbuff-1))*tiff_fields.ImageLength.value*tiff_fields.ImageWidth.value*tiff_fields.BitsPerSample.value/8 ...
            ,'UniformOutput',false)...
            ,'UniformOutput',false); 
        end
        
        j = cat(1,j{:})';% Format in order
        buffer([6*k;6*k+1],1:min(maxbuff,end_ind-start_ind+1-(i-1)*maxbuff))=j;
        
        j=cellfun(binfun...% Get the shorts
            ,arrayfun(@(x)dec2bin(x,32),...% Convert to binary
            header_length...
            +((i-1)*maxbuff+1:min(end_ind-start_ind+1,i*maxbuff))*(6+12*numel(tags))...The location of the next IFD
            ,'UniformOutput',false)...
            ,'UniformOutput',false);
        j = cat(1,j{:})';% Format in order
        buffer([end-1;end],1:min(maxbuff,end_ind-start_ind+1-(i-1)*maxbuff))=j;
        
        if i*maxbuff>=end_ind-start_ind+1% This contains the last IFD
            buffer([end-1;end], end_ind-start_ind+1-(i-1)*maxbuff)=0;% Label the last IFD with 0s
        end
        fwrite(O,buffer(:,1:min(maxbuff,end_ind-start_ind+1-(i-1)*maxbuff)),'uint16');
    end
    
else
    buffer = [buffer(1)+1;...
        254;4;0;1;0;0;...
        buffer(2:end)];    
    k = find(ismember(tags,'StripOffsets'))+1;
    j=dec2bin(tiff_fields.StripOffsets.value,32);
    j=binfun(j);
%     j=[bin2dec(j(17:32)) bin2dec(j(1:16))];
    buffer(6*k+[0 1])=j;
    buffer([end-1;end])=0;
    fwrite(O,buffer,'uint16');    
end

% Write the shared data
for i=1:numel(shared_inds)
    switch tiff_fields.(tags{shared_inds(i)}).sze% sze determines how to write
        case {1,2}% byte, ASCII
            fwrite(O,tiff_fields.(tags{shared_inds(i)}).value,'uint8');
        case 3% Short
            fwrite(O,tiff_fields.(tags{shared_inds(i)}).value,'uint16');
        case {4,5}% long, rational
            fwrite(O,tiff_fields.(tags{shared_inds(i)}).value,'uint32');
    end
end

if isequal(mode,'tiff42')&&StripsPerImage>1
    %%%%%% Writing the strip offsets, doing so in a buffer
    maxbuff = memory;
    maxbuff = min(2^(nextpow2((maxbuff.MaxPossibleArrayBytes-maxbuff.MemUsedMATLAB)/StripsPerImage)-5),end_ind-start_ind+1);
    buffer = header_length...
        +(end_ind-start_ind+1)*(6+12*numel(tags)+StripsPerImage*4)...The end of the IFDs & end of the strip data
        +shared_offsets(end)... The end of the shared data
        +(0:maxbuff*StripsPerImage-1)*tiff_fields.ImageWidth.value*tiff_fields.RowsPerStrip.value*tiff_fields.BitsPerSample.value/8;% Offsets between strips
    
    for i=1:ceil((end_ind-start_ind+1)/maxbuff)% For each buffer
        fwrite(O,buffer(1:min(end_ind-start_ind+1-(i-1)*maxbuff,maxbuff)*StripsPerImage),'uint32');% Write the remaining buffer
        buffer = buffer+maxbuff*tiff_fields.ImageLength.value*tiff_fields.ImageWidth.value*tiff_fields.BitsPerSample.value/8;% Increase the buffer for the next pass
    end
end

%%%%%%%%%% Writing the data
maxbuff = memory;% Ask available memory
maxbuff = min(maxbuff.MaxPossibleArrayBytes,max_write_buff);
maxbuff = min(2^(nextpow2(maxbuff/calcmult/8/tiff_fields.ImageWidth.value/tiff_fields.ImageLength.value)-3),end_ind-start_ind+1);

minn = inf;
maxx = -inf;
% keyboard
for i=1:ceil((end_ind-start_ind+1)/maxbuff)% For each buffer
    %     keyboard
%     disp(sprintf('%i/%i',i,ceil((end_ind-start_ind+1)/maxbuff)));
%     fclose(O);
%     O = fopen(outfile,'a');% Open the output file
%     pause(0.1);
    out = permute(data((i-1)*maxbuff+start_ind,min(i*maxbuff+start_ind-1,end_ind)),[2 1 3]); %Line changed for running motion correction
    fwrite(O,out,sprintf('uint%i',tiff_fields.BitsPerSample.value),endian);% Call data and write
    minn = min([out(:);minn]);
    maxx = max([out(:);maxx]);
    clear out;
%     pause(0.05);
end

if exist('ImageDescription','var')
    ImageDescription = sprintf([...
        'ImageJ=1.49k\n'...
        'images=%i\n'...
        'slices=%i\n'...
        'loop=false\n'...
        'min=%7.1f\n'...
        'max=%7.1f'...
        ],end_ind-start_ind+1,end_ind-start_ind+1,minn,maxx);
    fseek(O,...
        header_length...
    +((end_ind-start_ind+1)*(~isequal(mode,'imageJ_raw'))+isequal(mode,'imageJ_raw'))*(6+12*(numel(tags)+isequal(mode,'imageJ_raw')))...
        +sum(shared_offsets(1:find(shared_inds<find(ismember(tags,'ImageDescription')))))...
        ,-1);
    fwrite(O,ImageDescription,'uint8');
end

fclose(O);% Close the output file
% pause(5);