# ss-car-fusion
Information on the experiments for the CarFusion Dataset


--------------------------
### Preparing the dataset
The CarFusion Dataset consists of 10 sequences, which should be unzipped into the main dataset folder. The folder structure should be something like:

```
CarFusion-Dataset
    |
    |-car_craig1
    |     |- bb
    |     |- gt
    |     |- images_jpg
    |
    |-car_craig2
    .
    .
    .
```


---------
### To run LofTR matching evaluation on CarFusion
- Download the weights for LofTR from this [link](https://drive.google.com/drive/folders/1DOcOPZb3-5cWxLqn256AhwUVjBPifhuf?usp=sharing) and save it in the loftr/weights folder.
- `cd` into `loftr` folder.
- In the `match_carfusion_script.bat` or `.sh` file, change the `dataset_path` argument to the location of the CarFusion dataset folder .
---------