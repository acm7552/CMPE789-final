## Setup track on Andrew's device:

```cd D:\CARLA_0.9.15\WindowsNoEditor```

```conda activate carla15```

```CarlaUE4.exe```


## Set up town 03:

```cd D:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\util```
```python config.py --map Town03```


## Setting up vehicles and pedestrians

```cd D:\CARLA_0.9.15\WindowsNoEditor\PythonAPI\examples```

```python generate_traffic.py -n 50 -w 15```

```cd D:\CARLA_0.9.15\WindowsNoEditor\CMPE789-final``

```python spawnEgoVehicle.py```  


## Data
  
Expected directory structure:  
    output/  
    └── agent_x/  
        ├── rgb/         # RGB images  
        ├── seg/         # Colorized / processed segmentation (optional)  
        ├── seg_raw/     # CARLA raw segmentation masks  
        ├── depth/       # Processed / visualized depth maps (optional)  
        └── depth_raw/   # CARLA raw depth images  
  
## Run Training

```python seg_train_UNet_multiVehicle.py output```
```python depth_train_UNet_multiVehicle.py output```

Some optional arguments:
- `--epochs` (default: 30)
- `--batchSize` (default: 8)
- `--lr` (default: 1e-3)

## Outputs

- Model checkpoints: `unet_seg/`
- Loss curves and metrics: `seg_plots/`
- Validation predictions and confusion matrix

```python seg_train_UNet_multiVehicle.py output --epochs 30 --batchSize 8 --lr 1e-3```

```python depth_train_UNet_multiVehicle.py output --epochs 30 --batchSize 8 --lr 1e-3```



## Setup track on Yicheng's device:



