# Synthetic Resistor Generation

![](images/6.png)

This repo contains a python script `main.py` that uses blenderproc to render resistors for ML datasets. 

Running the script resusts in `bands.json` which is a list of the bands per image, and a folder `images/` which contains the rendered images. An example is provided.

### Pregenerated Datasets:
* 3000: https://huggingface.co/datasets/tylerebowers/synthetic_resistors

### Running the script

* Install dependencies: `pip install opencv-python blenderproc numpy`
* Download haven: `blenderproc download haven haven`, this downloads to a folder named "haven"
  * You dont need all of haven, but unfortunately you can't choose to just download hdri backgrounds, so wait for textures and backgrounds to finish then cancel the download. Or just download hdri backgrounds that you want manually.
* Run the script `blenderproc run main.py`
  * By default 1000 images are rendered, see the bottom of the script to set the number to generate.
  * Some combinations of `amdgpu` and hardware will sometimes crash randomly, I had to render on a different machine.
  * Each render takes 2-5 seconds.

### Notes
* To generate custom band orderings you can specify `order` in `create_procedural_resistor(order=None)` with a list of colors, such as `["brown", "black", "red", "gold"]`.
* There is a ton of usage from the `random` library, if you want to change varibility you will need to adjust these manually. 
