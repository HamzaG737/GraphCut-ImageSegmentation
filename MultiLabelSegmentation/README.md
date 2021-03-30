
# Multilabl segmentation

We consider this problem as an energy minimization problem, and we solve it using the alpha-beta-swap algorithm which relies on Graph cuts in its iterations. 

You can use as inputs RGB images. It will be converted it to grayscale.

The ```sh coords.py ``` should contain a contant COORDS that the user specify to denote the label regions chosen by user.

To run the  multi-label segmentation on an image and output the segmentation mask , please run :  
```sh
python main.py --ImgPath images/filename --n_labels
```
For example, for a 3-label segmentation, set n_label to 3.
The mask is saved in the images folder. We provide two input images as examples.

