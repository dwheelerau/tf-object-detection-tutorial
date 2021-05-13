# tf-object-detection-tutorial

This is based on the tutorial available here (https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/).  

The goal is to be able to detect rabit warrens in drone imagary, so the outcome of this work will be a workflow for processing drone footage.  At this stage I think the workflow would look like this:  

- Downsize video via dropping frames (optional)  
- Reduce size of all frames (optional)  
- Processing  
  - run model across frames and predict objects  
  - report positive frames in a summary spreadsheet  
  - rejoin frames with boundary boxes drawn around objects  
- training  
  - Manually identify positive frames to train on  
  - Annotate the frames using labellmg (https://tzutalin.github.io/labelImg/)  
  - split the data into training and test sets  
  - convert xml and images to tf records  
  - train the model (ie this tutorial)  
  - assess the model with the test set  

For this work I will train on cats just to ensure that the model works on a single category.  I then will use the same pipeline to train on warrens.  

## Tutorial on cats  

The cats dataset was part of the xx package, I am only going to annotate xxx, I will follow the workflow here but I also should try it with the MD model which should be a good starting point for cat identification as well as perhaps warrens given they glow in the dark of the IR cameras used on these drones.  


