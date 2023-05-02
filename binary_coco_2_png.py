#!pip install pycocotools
#!pip install pillow
#Usage python binary_coco_2_png.py --json "folder/result.json" --output_folder "folder/masks/"
#Generate masks for binary image segmentation image from a coco json.


from pycocotools.coco import COCO
from PIL import Image
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description='Turning COCO json of binary masks into PNGs.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-j', '--json',   type=str, help='The Coco json file', dest='json', required=True)
    parser.add_argument('-o', '--output_folder',   type=str, help='The output folder of the images', dest='output_folder', required=True)
    return parser.parse_args()
 
if __name__ == '__main__':

    args = get_args()
    coco = COCO(args.json)
    
    #Making sure the masks of the images are binary
    try:
        cat_ids = coco.getCatIds()
        if len(cat_ids) != 1:
            raise InterruptedError
    except Exception :
        exit(f" The masks are not binary")
        
    #Making sure the extensions is png or jpg
    try:
        if args.ext != "jpg" and args.ext != "png":
            raise InterruptedError
    except Exception :
        exit(f"The extensions can only be png or jpg")         
    
    #Generate masks for each image
    for _,  img in coco.imgs.items():
        #Getting the mask
        anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)
        #Stacking the masks for each image
        mask = coco.annToMask(anns[0])
        for i in range(len(anns)):
            mask += coco.annToMask(anns[i])
        # Creating the file
        im = Image.fromarray(mask*255)
        name_without_ext = (Path((img["file_name"])).parts[-1].split(".")[0])
        name = Path(args.output_folder,f"{name_without_ext}.png")
        im.save(name)
     
        
