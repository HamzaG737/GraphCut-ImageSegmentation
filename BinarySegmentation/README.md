
# Binary segmentation

This is repo for binary segmentation using graph-cuts. The algorithms that we implemented to find max-flow are : Ford-Fulkerson , Push-Relabel and Boykov-Kolmogorov. By default in the code we use Ford-Fulkerson. 

You can use as inputs RGB images , without constraints on the shape. Note however that we will resize the image to $$30 \times 30$$ shape , and convert it to grayscale. So to get good results it is better to provide image with low resolution and already in grayscale format. 

To run the graph cut algorithm and output the segmentation mask , please run :  
```sh
python main.py --ImgPath images/filename
```
The mask is saved in the images folder. We provide two input images as examples.
You can change the resizing parameter --resize_factor if you want. But increasing it may result in a long computation time. 