## SILK code documentation
- To prepare silk for refer to this page: [Silk Setup](https://github.com/facebookresearch/silk/blob/main/doc/usage/setup.md)
#### To run silk inference on a pair of images
- The script to run is `./examples/silk-inference.py.`
- To change the input images: replace `IMAGE_0_PATH` and `IMAGE_1_PATH`.
- I copied the script to the main project folder to run it. I also copied `common.py` to it.


  - 
### common.py ENVIRONMENT VARIABLES
- `CHECKPOINT_PATH (Path to model checkpoint).`
- `SILK_NMS (Non-maximum Suppression).`
- `SILK_THRESHOLD (Keypoint Score Threshold).`
- `SILK_TOP_K` (Top-K Descriptors).
- `SILK_SCALE_FACTOR` (Scaling of Descriptor output)
- `SILK_BACKBONE` (ParametricVGG)
- `SILK_MATCHER` (options are `ratio-test`, `double-softmax`)
#### silk.py documentation
- **Main classes**: `SiLKLoFTR` , `SiLKVGG`, `SiLKBase`.
- **Additional Functions**:  `from_feature_coords_to_image_coords(model, desc_positions)`.
- **Imports**
  - `silk.flow` import `AutoForward`, `Flow`

### utils.py 

- `logits_to_prob` (Converts logits to prob)
- `depth_to_space` (**TODO**)
- `prob_map_to_points_map` (**TODO**)
- `remove_border_points` (Remove border points)
- `original_nms` (NMS)
- `fast_nms` (NMS)
- `space_to_depth` (**TODO**)
- `positions_to_label_map` (**TODO**)
- `float_positions_to_int` (**TODO**)
- `


##### (Class) SiLKBase(AutoForward, torch.nn.Module)
- `(Function)` **Constructor**
  - `self.backbone = SharedBackboneMultipleHeads`
  - `self.detector_heads = set()`
  - `self.descriptor_heads = set()`
- `(Property)` `coordinate_mapping_composer`
  -  `return self.backbone.coordinate_mapping_composer`
- `(Function)` `add_detector_head(self, head_name, head, backbone_output_name=None):`
   - `self.backbone.add_head_to_backbone_output(head_name, head, backbone_output_name)`
   - `self.detector_heads.add(head_name)`
- `(Function)` `add_descriptor_head(self, head_name, head, backbone_output_name=None):`
  - `self.backbone.add_head_to_backbone_output(head_name, head, backbone_output_name)`
  - `self.descriptor_heads.add(head_name)`

##### (Class) SiLKVGG(SiLKBase):
- **Constructor**
     - `backbone = (VGGBackbone(...))`
     - `detector_head = (VGGDetectorHead(...))`
     - `descriptor_head = (VGGDescriptorHead(...))`
     - `self.add_detector_head("logits", detector_head)`
     - `self.add_descriptor_head("raw_descriptors", descriptor_head)`
     - `self.descriptor_scale_factor = nn.parameter.Parameter(torch.tensor(descriptor_scale_factor),requires_grad=learnable_descriptor_scale_factor)`
     - `self.normalize_descriptors = normalize_descriptors`
     - `MagicPoint.add_detector_head_post_processing(...)`
     - `SiLKVGG.add_descriptor_head_post_processing(...)`
- `(Function)` `SiLKVGG.add_descriptor_head_post_processing(`
   - `self.flow,`    #  Not sure exactly 
   - `input_name=self.backbone.input_name,` #  define the inputs
   - `descriptor_head_output_name="raw_descriptors",` # define the 
   - `scale_factor=self.descriptor_scale_factor,`
   - `normalize_descriptors=normalize_descriptors,)`
- `(Function)` `get_dense_positions(probability)`
- `(Function)` `get_dense_descriptors(normalized_descriptors)`
- `(Function)` `sparsify_descriptors(...)`  

##### (Class) SiLKLofTR(SiLKBase):
- **Constructor**
    - `backbone = (ResNetFPN_8_2(...))`
    - `detector_head = (VGGDetectorHead(...))`
    - `descriptor_head = (VGGDescriptorHead(...))`
    - `self.add_detector_head(...)` *logit descriptor* `output is features`
    - `self.add_descriptor_head(...)` *raw descriptors* `output is features`
    -  `MagicPoint.add_detector_head_post_processing`
    -  `SiLKVGG.add_descriptor_head_post_processing`


