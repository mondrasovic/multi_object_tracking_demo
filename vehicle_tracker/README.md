# Vehicle Tracker

## Running the script

```
python3 ../model/yolov3.cfg ../model/yolov3.weights ../model/coco.names -i ../data/car_sample.mp4
```

---

## Custom runs

### File **01.mp4**

```
python3 ../model/yolov3.cfg ../model/yolov3.weights ../model/coco.names -i ../data/01.mp4 -s 0.5 -n 0.3
```