# Face classification (binary) from an image
Computer vision project to detect and classify the faces extracted from an image.
This is a binary classifier, that detects whether the person is 'RIZWAN' (me) or 'OTHER'.

#### Neural Schema - CNN
```
ConvNet(
  (layer1): Sequential(
    (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): ReLU()
  )
  (layer2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): ReLU()
  )
  (drop_out): Dropout(p=0.5, inplace=False)
  (fc1): Sequential(
    (0): Linear(in_features=4096, out_features=128, bias=True)
    (1): ReLU()
  )
  (fc2): Linear(in_features=128, out_features=2, bias=True)
)
(Pdb) 
```

#### Model Validation
```
[2021-08-31 00:39:08,056] [generate_data.py:26] - INFO - Dataset sample length: 2098
[2021-08-31 00:39:08,057] [generate_data.py:38] - INFO - Train data size: 1783
[2021-08-31 00:39:08,057] [generate_data.py:39] - INFO - Test data size: 315
[2021-08-31 00:39:08,057] [generate_data.py:46] - INFO - Train generator size: 112
[2021-08-31 00:39:08,057] [generate_data.py:47] - INFO - Test generator size: 20
[2021-08-31 00:39:08,057] [train_validate.py:38] - INFO - Validating the model - cuda:0
[2021-08-31 00:39:08,057] [train_validate.py:30] - INFO - Loading the model - cuda:0
[2021-08-31 00:39:09,714] [train_validate.py:55] - INFO - Accuracy of Test Data: 99.37%
```

#### Model test with an Image
```
[2021-08-31 00:32:37,702] [train_validate.py:31] - INFO - Loading the model - cuda:0
[2021-08-31 00:32:39,197] [image_processor.py:80] - INFO - Extracting faces
[2021-08-31 00:32:39,336] [image_processor.py:25] - INFO - Found [4] faces
[2021-08-31 00:32:39,354] [image_processor.py:96] - INFO - Detected images of : ['OTHER', 'RIZWAN', 'OTHER', 'OTHER']
```
