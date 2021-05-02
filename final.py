import pixellib
from pixellib.instance import instance_segmentation

segment_image = instance_segmentation() # 1 second
segment_image.load_model("mask_rcnn_coco.h5")
segment_image.segmentImage("sample6.jpg", output_image_name= "img_new6.jpg", show_bboxes= True)

segment_image_avg = instance_segmentation(infer_speed= "average") # 0.5 seconds
segment_image_avg.load_model("mask_rcnn_coco.h5") 
segment_image_avg.segmentImage("sample6.jpg", output_image_name = "img_new6_avg.jpg", show_bboxes= True)

segment_image_fast = instance_segmentation(infer_speed= "fast") # 0.35 seconds
segment_image_fast.load_model("mask_rcnn_coco.h5") 
segment_image_fast.segmentImage("sample6.jpg", output_image_name = "img_new6_fast.jpg", show_bboxes= True)

segment_image_rapid = instance_segmentation(infer_speed= "rapid") # 0.2 seconds
segment_image_rapid.load_model("mask_rcnn_coco.h5") 
segment_image_rapid.segmentImage("sample6.jpg", output_image_name = "img_new6_rapid.jpg", show_bboxes= True)
