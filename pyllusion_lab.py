# imports & set up
from IPython.display import display, clear_output
from IPython.display import Image as IPyImage
import pandas as pd
import pyllusion
from ipywidgets import IntSlider, FloatSlider, VBox, HBox, Output, Button, Text, Layout
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random

# Reusable utility functions
def save_results_to_csv(results, results_filename="../expt_results/results.csv"):
    """save the results dataframe to results_filename"""
    results.to_csv(results_filename, index=False, header=True)
    print(f"Saved {len(results)} rows to {results_filename}")

def plot_adjustment_results(adjustment_results_df, results_fig_filename):
    """plot summary of your results for across multiple illusion strengths (arrow angles)"""
    adjustment_results_df.plot('illusion_strength','PSE', marker='o', linestyle='-')
    plt.xlabel('Illusion strength')
    plt.ylabel('Extra comparison size needed\n to be perceived as same as standard')
    plt.title('Method of Adjustment')
    plt.tight_layout()
    plt.savefig(results_fig_filename) 
    print("Saved figure to " + results_fig_filename)
    plt.show()

def pil_to_ipyimage(pil_img, scale=1.0):
    """Convert a Pillow image to an IpyImage and also optionally resize it"""
    w, h = pil_img.size
    new_size = (int(w * scale), int(h * scale))
    resized = pil_img.resize(new_size, resample=Image.LANCZOS)

    buf = BytesIO()
    resized.save(buf, format='PNG')
    return IPyImage(data=buf.getvalue())

def pyllusion_adjustment_expt(illusion_type, illusion_strength=30, standard=0.5, instructions="", slider_min=0.0, slider_max=1.0, img_scale=0.25):
    """An simple illusion perception experiment using the Method of Adjustment and a pyllusion-generated stimulus.
    The observer adjusts a slider until the two stimuli look the same, and then presses the Submit button"""

    # make sure the illusion_type is available
    avail_pyllusions = [name for name in dir(pyllusion) if name[0].isupper()]
    if illusion_type not in avail_pyllusions:
        print(illusion_type + " is not one of pyllusion's available illusions: ")
        print(avail_pyllusions)
        return
        
    # Define a frame update "event handler" function that redraws the stimulus image
    def update_image(change):
        delta = change['new'] # this is the slider's value
        # generate the pyllusion stimulus (change this line if you want to generate the stimulus a different way)
        #stimulus = eval("pyllusion." + illusion_type + "(illusion_strength=" + str(illusion_strength) + ", size_min=" + str(standard) + ", difference=" + str(delta) + ")").to_image()
        with output:
            output.clear_output(wait=True)
            img = eval(f"pyllusion.{illusion_type}(illusion_strength={illusion_strength}, size_min={standard}, difference={delta})").to_image()
            display(pil_to_ipyimage(img, scale=img_scale))
    
    # Define the handler for the submit button 
    def on_submit_clicked(b):
        adjustment_results['illusion_strength'] = illusion_strength
        adjustment_results['difference'] = adjustment_slider.value
        adjustment_results['standard'] = standard 
        adjustment_results['comparison'] = standard + standard*adjustment_results['difference'] # this formula is specific to pyllusion's use of the difference parameter
        adjustment_results['PSE'] = adjustment_results['comparison'] - adjustment_results['standard']
        with output:
            print(f"The standard has a size of {adjustment_results['standard']:.3f}")
            print(f"You adjusted the comparison to a size of {adjustment_results['comparison']:.3f}")
            print(f"This means that when the illusion_strength parameter is {illusion_strength},")
            print(f"your Point of Subjective Equality (PSE) is {adjustment_results['PSE']:.3f}")
        adjustment_slider.disabled = True # disable further interaction
        submit_button.disabled = True # disable further interaction
        return adjustment_results

    # Print instructions and set up the components
    print(instructions)
    print("When you are done, hit Submit to see you results.")
    output = Output() # Set up the output area for the image
    centered_output = HBox([output], layout=Layout(justify_content='center'))
    adjustment_slider = FloatSlider(value=random.uniform(slider_min, slider_max), readout=False, # Create the slider to adjust line length difference
        min=slider_min, max=slider_max, step=(slider_max-slider_min)/20.0, description="size:",continuous_update=False)
    centered_slider = HBox([adjustment_slider], layout=Layout(justify_content='center'))
    submit_button = Button(description="Submit", button_style="success") # Submit button
    centered_button = HBox([submit_button], layout=Layout(justify_content='center'))
    adjustment_results = {}  # to store submission result
  
    # Attach event handlers & display
    adjustment_slider.observe(update_image, names='value')
    submit_button.on_click(on_submit_clicked)
    update_image({'new': adjustment_slider.value}) # Trigger initial draw manually
    display(VBox([centered_slider, centered_output, centered_button], layout=Layout(align_items='center')))

    return adjustment_results