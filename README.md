PRESENTATION:
https://docs.google.com/presentation/d/1GsoWhhQTyWRvnEiNOnKWoI57yluuJboJEqWFv4XM-Zs/edit?usp=sharing

AI-Powered System for Optimizing Speed Limits and Congestion

Purpose and Overview:

Looks at various features (speed limit and average historical speed of road), extracted from the SegmentSpeed Inrix API.
Uses a neural network model to calculate the road congestion based on these two features
The model allows a city planner to determine the optimal speed limit for the least road congestion.
Methods:

We first extracted data from the Inrix SegmentSpeed API and created a dataset using the Pandas library.
Trained the data on the 20 percent of the data set to fit the model.
Tested on various speed limits, while keeping historical average constant.
Analyzed predicted congestion values.
