%{
# Area mask for each scan
-> experiment.Scan
-> anatomy.Area
-> shared.Field
---
-> map.RetMap
mask                     : mediumblob            # mask of area
%}

classdef AreaMask < dj.Manual
    methods
        function createMasks(obj, key, varargin)
            
            params.exp = 1.5;
            params.sigma = 2;
            
            params = ne7.mat.getParams(params,varargin);
            
            if strcmp(fetch1(experiment.Scan & key,'aim'),'widefield')
                contiguous = 1;
            else
                contiguous = 0;
            end
            % populate if retinotopy map doesn't exist
            ret_key = getRetKey(map.RetMap, key);
            
            % get maps
            background = getBackground(map.RetMap & ret_key, params);
            
            % if FieldCoordinates exists add it to the background
            ref_key = fetch(anatomy.RefMap & (map.RetMapScan & key));
            if exists(anatomy.FieldCoordinates & ref_key)
                background = cat(4,background,plot(anatomy.FieldCoordinates & ref_key));
            end
            
            % get masks already extracted
            if exists(obj & rmfield(key,'ret_idx'))
                [area_map, keys] = getContiguousMask(obj, rmfield(key,'ret_idx'),contiguous);
            else
                area_map = zeros(size(background,1),size(background,2));
            end
            
            % create masks
            area_map = ne7.ui.paintMasks(abs(background),area_map);
            if isempty(area_map); disp 'No masks created!'; return; end
            
            % delete previous keys if existed
            if exists(obj & rmfield(key,'ret_idx'))
                del(anatomy.AreaMask & keys)
            end
            
            % image
            figure;
            masks = ne7.mat.normalize(area_map);
            masks(:,:,2) = 0.2*(area_map>0);
            masks(:,:,3) = background(:,:,1,1);
            ih = image(hsv2rgb(masks));
            axis image
            axis off
            shg
            
            % loop through all areas get area name and insert
            areas = unique(area_map(:));
            areas = areas(areas>0);
            brain_areas = [];
            for iarea = areas'
                % fix saturation for selected area
                colors =  0.2*(area_map>0);
                colors(area_map==iarea) = 1;
                masks(:,:,2) = colors;
                ih.CData = hsv2rgb(masks);
                s = regionprops(area_map==iarea,'area','Centroid');
                th = text(s(1).Centroid(1),s(1).Centroid(2),'?');
                shg
                
                % ask for area name
                areas = fetchn(anatomy.Area,'brain_area');
                area_idx = listdlg('PromptString','Which area is this?',...
                    'SelectionMode','single','ListString',areas);
                brain_areas{iarea} = areas{area_idx};
                th.delete;
                
                % set correct area label
                text(s(1).Centroid(1),s(1).Centroid(2),brain_areas{iarea})
            end
            
            % get base key
            tuple = ret_key;
            
            if ~contiguous
                % get field specific area map
                [field_area_maps, fields] = splitContiguousMask(obj, tuple, area_map);
            else
                field_area_maps{1} = area_map;
                fields(1) = 1;
            end
            
            % loop through all fields
            for ifield = 1:length(fields)
                tuple.field = fields(ifield);
                
                % loop through all areas get area name and insert
                areas = unique(field_area_maps{ifield}(:));
                areas = areas(areas>0);
                for iarea = areas'
                    
                    % get area name
                    tuple.brain_area = brain_areas{iarea};
                    tuple.mask = field_area_maps{ifield} == iarea;
                    insert(obj,tuple)
                end
            end
        end
        
        function extractMasks(obj, keyI)
            
            % fetch all area masks
            map_keys = fetch(anatomy.AreaMask & (anatomy.RefMap & (proj(anatomy.RefMap) & (anatomy.FieldCoordinates & keyI))));
            
            % loop through all masks
            for map_key = map_keys'
                [mask, area, ret_idx] = fetch1(anatomy.AreaMask & map_key, 'mask', 'brain_area', 'ret_idx');
                
                % loop through all fields
                for field_key = fetch(anatomy.FieldCoordinates & keyI)'
                    
                    % find corresponding mask area
                    fmask = filterMask(anatomy.FieldCoordinates & field_key, mask);
                    
                    % insert if overlap exists
                    if ~all(~fmask(:))
                        tuple = rmfield(field_key,'ref_idx');
                        tuple.brain_area = area;
                        tuple.mask = fmask;
                        tuple.ret_idx = ret_idx;
                        insert(obj,tuple)
                    end
                end
            end
        end
        
        function [fmasks, fields] = splitContiguousMask(~, key, ref_mask)
            
            % fetch images
            if strcmp(fetch1(experiment.Session & key,'rig'),'2P4')
                [x_pos, y_pos, fieldWidths, fieldHeights, fieldWidthsInMicrons,keys] = ...
                    fetchn(meso.ScanInfoField * meso.SummaryImagesAverage & key,...
                    'x','y','px_width','px_height','um_width');
                
                % calculate initial scale
                pxpitch = mean(fieldWidths.\fieldWidthsInMicrons);
                
                % start indexes
                XX = (x_pos - min(x_pos))/pxpitch;
                YY = (y_pos - min(y_pos))/pxpitch;
                
                % deconstruct the big field of view
                for ifield = 1:length(x_pos)
                    fields(ifield) = keys(ifield).field;
                    fmasks{ifield} = ref_mask(YY(ifield)+1:fieldHeights(ifield)+YY(ifield),...
                        XX(ifield)+1:fieldWidths(ifield)+XX(ifield));
                end
            else % for all other scans there is no need to split the mask
                keys = fetch(meso.ScanInfoField * reso.SummaryImagesAverage & key);
                for ikey = 1:length(keys)
                    fields(ikey) = keys(ikey).field;
                    fmasks{ikey} = ref_mask;
                end
            end
        end
        
        function [area_map, keys, background] = getContiguousMask(obj, key, override)
            
            % fetch masks & keys
            [masks, keys] = fetchn(obj & key,'mask');
            
            % get information from the scans depending on the setup
            if (nargin<3 || ~override) && (strcmp(fetch1(experiment.Session & key,'rig'),'2P4') || length(masks)<2)
                [x_pos, y_pos, fieldWidths, fieldHeights, fieldWidthsInMicrons, masks, areas, avg_image] = ...
                    fetchn(obj * meso.ScanInfoField * meso.SummaryImagesAverage & key,...
                    'x','y','px_width','px_height','um_width','mask','brain_area','average_image');
                
                % calculate initial scale
                pxpitch = mean(fieldWidths.\fieldWidthsInMicrons);
                
                % construct a big field of view
                x_pos = (x_pos - min(x_pos))/pxpitch;
                y_pos = (y_pos - min(y_pos))/pxpitch;
                area_map = zeros(ceil(max(y_pos+fieldHeights)),ceil(max(x_pos+fieldWidths)));
                background = zeros(size(area_map));
                for islice =length(masks):-1:1
                    mask = double(masks{islice})*find(strcmp(areas{islice},unique(areas)));
                    y_idx = ceil(y_pos(islice)+1):ceil(y_pos(islice))+size(mask,1);
                    x_idx = ceil(x_pos(islice)+1):ceil(x_pos(islice))+size(mask,2);
                    back = area_map(y_idx, x_idx);
                    area_map(y_idx, x_idx) = max(cat(3,mask,back),[],3);
                    background(y_idx, x_idx) = avg_image{islice};
                end
                
            else
                area_map = zeros(size(masks{1}));
                for imasks = 1:length(masks)
                    area_map(masks{imasks}) = imasks;
                end
                background = [];
            end
        end
        
        function plot(obj, varargin)
            
            params.back_idx = [];
            params.exp = 0.4;
            
            params = ne7.mat.getParams(params,varargin);
            
            if strcmp(fetch1(proj(experiment.Scan,'aim') & obj,'aim'),'widefield')
                contiguous = 1;
            else
                contiguous = 0;
            end
            
            % get masks
            [area_map, keys, mask_background] = getContiguousMask(obj,fetch(obj),contiguous);
            areas = {keys(:).brain_area}';
            
            % get maps
            if exists(map.RetMap & (map.RetMapScan &  obj))
                background = getBackground(map.RetMap & (map.RetMapScan &  obj));
                
                % if FieldCoordinates exists add it to the background
                if exists(anatomy.FieldCoordinates & proj(anatomy.RefMap & obj))
                    background = cat(4,background,plot(anatomy.FieldCoordinates & ...
                        proj(anatomy.RefMap & obj)));
                    if isempty(params.back_idx)
                        params.back_idx = size(background,4);
                    end
                end
            else
                background = mask_background;
            end
            
            % adjust background contrast
            background = ne7.mat.normalize(abs(background.^ params.exp));
            
            % merge masks with background
            sat = background(:,:,1,1);
            sat(area_map==0) = 0;
            im = hsv2rgb(cat(3,ne7.mat.normalize(area_map),sat,background(:,:,1,1)));
            if nargin<2 || isempty(params.back_idx) || params.back_idx > size(background,4)
                image((im));
            else
                imshowpair(im,background(:,:,:,params.back_idx),'blend')
            end
            axis image;
            key = fetch(proj(experiment.Scan) & obj);
            set(gcf,'name',sprintf('Animal:%d Session:%d Scan:%d',key.animal_id,key.session,key.scan_idx))
            
            % place area labels
            un_areas = unique(areas);
            for iarea = 1:length(un_areas)
                s = regionprops(area_map==iarea,'Centroid');
                text(s(1).Centroid(1),s(1).Centroid(2),un_areas{iarea},'color',[1 0 0],'fontsize',16)
            end
        end
        
        
    end
end