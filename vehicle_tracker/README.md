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

### Re-identification evaluator

```
../model/embedding_model_layout.json ../model/embedding_model_weights.h5 ../../../datasets/VeRi/VeRi_with_plate/image_test ../../../datasets/VeRi/VeRi_with_plate/name_test.txt ../../../datasets/VeRi/VeRi_with_plate/name_query.txt ../../../datasets/VeRi/VeRi_with_plate/gt_index.txt ../../../datasets/VeRi/VeRi_with_plate/jk_index../model/embedding_model_layout.json ../model/embedding_model_weights.h5 ../../../datasets/VeRi/VeRi_with_plate/image_test ../../../datasets/VeRi/VeRi_with_plate/name_test.txt ../../../datasets/VeRi/VeRi_with_plate/name_query.txt ../../../datasets/VeRi/VeRi_with_plate/gt_index.txt ../../../datasets/VeRi/VeRi_with_plate/jk_index.txt
```
