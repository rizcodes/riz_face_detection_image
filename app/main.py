#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from settings import project_dirs
from model_trainer.generate_data import GenerateData
from model_trainer.train_validate import TrainValidateModel, device
from app.image_processor import ProcessImage

# Generate dataset
generate_data = GenerateData(data_dir=project_dirs['data_dir'], split_ratio=0.85)
train_generator, test_generator = generate_data.load_generators()

# Train and Validate model
tvm = TrainValidateModel()
tvm.train_dataset(dataset=train_generator)
tvm.validate_dataset(dataset=test_generator)

# Test - unseen data
img = ProcessImage(image_path=os.path.join(project_dirs['proj_dir'], 'bakrid2019.jpg'))
img.detect_person(net=tvm.load_model(), device=device)
