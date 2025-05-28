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
import os
import time
from datetime import datetime

# Reusable utility functions
def save_results_to_csv(results, results_filename="../expt_results/results.csv"):
    """save the results dataframe to results_filename"""
    results.to_csv(results_filename, index=False, header=True)
    print(f"Saved {len(results)} rows to {results_filename}")


def pil_to_ipyimage(pil_img, scale=1.0):
    """Convert a Pillow image to an IpyImage and also optionally resize it"""
    w, h = pil_img.size
    new_size = (int(w * scale), int(h * scale))
    resized = pil_img.resize(new_size, resample=Image.LANCZOS)

    buf = BytesIO()
    resized.save(buf, format='PNG')
    return IPyImage(data=buf.getvalue())


def check_illusion_type(illusion_type):
    """Return True if illustion_type is in the pyllusion package"""
    avail_pyllusions = [name for name in dir(pyllusion) if name[0].isupper()]
    if illusion_type not in avail_pyllusions:
        print(illusion_type + " is not one of pyllusion's available illusions: ")
        print(avail_pyllusions)
        return False
    else:
        return True

def render_illusion(illusion_type, illusion_strength, standard, difference):
    """Render a pyllusion illusion, return it as a Pillow image and its parameters"""
    illusion = eval(f"pyllusion.{illusion_type}(illusion_strength={illusion_strength}, size_min={standard}, difference={difference})")
    return illusion.to_image(), illusion.get_parameters() 

def pre_render_stimuli(illusion_type, illusion_strengths, standard, differences, 
                       prerendered_stimuli_dir="../expt_results/pre_rendered_stimuli/"):
    """
    Pre-render and save stimuli for a set of illusion_strengths and set of delta values.

    Parameters
    ----------
    illusion_type : str
        Name of the illusion, e.g. "MullerLyer"
    illusion_strengths : list of floats
        Strength parameter for the illusion
    standard : float
        Baseline size parameter
    differences : list of floats
        List of difference values to render
    prerendered_stimuli_dir : str
        Directory in which to save PNG files

    Returns
    -------
    stimuli_dict : dict
        Mapping from delta -> {'image_path': str, 'params': dict}
    """
    os.makedirs(prerendered_stimuli_dir, exist_ok=True) # Create output directory if it doesn't exist
    
    stimuli_dict = {}
    for strength in illusion_strengths:
        stimuli_dict[strength] = {} 
        for delta in differences:
            print(strength, delta)
            # Render the illusion once
            stimulus, illusion_params = render_illusion(illusion_type, strength, standard, delta)
            
            # Save the image
            filename = f"{illusion_type}_strength{strength}_std{standard}_delta{delta}.png"
            filepath = os.path.join(prerendered_stimuli_dir, filename)
            stimulus.save(filepath)
            
            # Store path and params
            stimuli_dict[strength][delta] = {
                "image_path": filepath,
                "params": illusion_params
            }    
    print(str(len(illusion_strengths)*len(differences)) + " illusion stimuli written to " + prerendered_stimuli_dir)
    return stimuli_dict

def load_prerendered_stimulus(stimuli_dict, illusion_strength=30, difference=0):
    """Load a pre-rendered stimulus, returns the Pillow image and the pyllusion parameters"""
    if os.path.exists(stimuli_dict[illusion_strength][difference]["image_path"]):
        stimulus_img = Image.open(stimuli_dict[illusion_strength][difference]["image_path"])
        stimulus_params = stimuli_dict[illusion_strength][difference]["params"]
        return stimulus_img, stimulus_params
    else: 
        return None, None

def pyllusion_adjustment_expt(illusion_type, illusion_strength=30, standard=0.5, instructions="", slider_min=0.0, slider_max=1.0, img_scale=0.25):
    """An simple illusion perception experiment using the Method of Adjustment and a pyllusion-generated stimulus.
    The observer adjusts a slider until the two stimuli look the same, and then presses the Submit button"""

    # make sure the illusion_type is available
    if not check_illusion_type(illusion_type):
        return
        
    # Define a frame update "event handler" function that redraws the stimulus image
    def update_image(change):
        delta = change['new'] # this is the slider's value
        # generate the pyllusion stimulus (change this line if you want to generate the stimulus a different way)
        #stimulus = eval("pyllusion." + illusion_type + "(illusion_strength=" + str(illusion_strength) + ", size_min=" + str(standard) + ", difference=" + str(delta) + ")").to_image()
        with output:
            output.clear_output(wait=True)
            img, illusion_params = render_illusion(illusion_type, illusion_strength, standard, delta)
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

def centered_text(content):
    display(HTML(f"<div style='text-align: center;'>{content}</div>"))

def save_results_to_csv(df, results_filename):
    df.to_csv(results_filename, index=False)
    print(f"Results saved to {results_filename}")

def pyllusion_constantstim_expt(illusion_type, illusion_strength=30, differences=[-1, -0.5, 0, 0.5, 1], num_trials_per_level=2, size_min=0.5,
                                duration=0.8, output_data_path="../expt_results/", welcome_instructions="Welcome!", trial_instructions="Which looked bigger?", 
                                stimuli_dict={}, img_scale=0.5):
    """An simple illusion perception experiment using the Method of Constanst Stimuli and a pyllusion-generated stimulus.
    The observer clicks one of two buttons on repeated trials in random order"""

    # make sure the illusion_type is available
    if not check_illusion_type(illusion_type):
        return

    # setup UX
    messages = Output()
    image_box = Output(layout=Layout(display='flex', justify_content='center')) # a centered box for the illusion stimulus, which starts empty
    controls = VBox() 
    main_ui = VBox([image_box, messages, controls]) # a centered box with the illusion, the messages and controls
    display(main_ui)
    observer_id_input = Text(description="Observer ID:", placeholder="e.g. 001")
    start_button = Button(description="Start Experiment", button_style='success')
    controls.children = [observer_id_input, start_button]
    with messages:
        messages.clear_output(wait=True)
        print(welcome_instructions)    

    # set up trials using method of constant stimuli
    deltas = np.random.permutation(differences * num_trials_per_level) # randomized order of stimuli
    results_df = pd.DataFrame(columns=['trial', 'illusion_strength', 'standard', 'difference', 'response', 'RT', 'size1', 'size2', 'standard1', 'chooseComparison'])
    trial_data = {"i": 0, "start_time": None, "results": results_df, "standard1": None, "size1": None, "size2": None}
    results_filename = output_data_path + "temp.csv"    

    def show_buttons():
        """Show two response buttons"""        
        # 1) Erase stimulus
        image_box.clear_output(wait=False)
        
        # 2) Put trial instructions into the output widget
        with messages:
            messages.clear_output(wait=True)
            print(f"Trial {trial_data['i']+1}/{len(deltas)} — {trial_instructions}")
    
        # 3) Show the two choice buttons
        controls.children = [VBox([top_button, bottom_button],
                layout=Layout(align_items='center', justify_content='center')
            )
        ]
    
    def run_trial():
        """run one trial"""
        i = trial_data["i"]        
        if i >= len(deltas): # end of experiment
            with messages:
                messages.clear_output(wait=True)
                print("Experiment complete. Data saved to " + results_filename)
            controls.children = []
            save_results_to_csv(trial_data["results"], results_filename=results_filename)
            return
            
        # load the pre-generated stimulus
        stimulus_img, stimulus_params = load_prerendered_stimulus(stimuli_dict, illusion_strength=illusion_strength, difference=deltas[i])
        trial_data["size1"] = stimulus_params['Size_Top']
        trial_data["size2"] = stimulus_params['Size_Bottom']
        if illusion_type == "MullerLyer":
           trial_data["standard1"] = stimulus_params['Distractor_TopLeft1_x1'] > stimulus_params['Distractor_TopLeft1_x2'] # ML specific

        # Show the stimulus
        image_box.clear_output(wait=True)
        with image_box:
            display(pil_to_ipyimage(stimulus_img, scale=img_scale))
       
        trial_data["start_time"] = time.time()
        time.sleep(duration)
        show_buttons()
        trial_data["i"] += 1

    def record_response(response_code):
        """Called everytime a response button is clicked"""
        rt = time.time() - trial_data["start_time"]
        i = trial_data["i"]
        results_df = trial_data["results"]
        if illusion_type == "MullerLyer":
            # choseTop is True if observer chose the top stimulus
            chooseTop = np.where(response_code==1, True, False)
            # chooseComparison is True if only ONE of standard1 and chooseTop is True, otherwise False
            chooseComparison = np.logical_xor(trial_data["standard1"], chooseTop)
        else:
            chooseComparison = None
        results_df.loc[len(results_df)] = [i, illusion_strength, size_min, deltas[i - 1], response_code, rt, trial_data["size1"], trial_data["size2"], trial_data["standard1"], chooseComparison]                
        trial_data["results"] = results_df

        # Clear and proceed with the next trial
        image_box.clear_output(wait=True)
        messages.clear_output(wait=True)
        controls.children = []
        run_trial()

    def on_start_clicked(b):
        """Call once, when the start button is clicked"""
        nonlocal results_filename
        observer_id = observer_id_input.value.strip()
        with messages:
            if not observer_id:
                messages.clear_output(wait=True)
                print("Please enter an Observer ID before starting.")
                return
            else:
                messages.clear_output(wait=False) # clear the welcome instructions
        controls.children = [] # clear the start button
        results_filename = f"{output_data_path}{illusion_type}_{illusion_strength}_constant_stimuli_{observer_id}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}_results.csv"
        run_trial()

    # button set up and callbacks
    start_button.on_click(on_start_clicked)
    top_button = Button(description="Top (or Left) Bigger", button_style='info', layout=Layout(width="400px"))
    bottom_button = Button(description="Bottom (or Right) Bigger", button_style='info', layout=Layout(width="400px"))
    top_button.on_click(lambda b: record_response("1"))
    bottom_button.on_click(lambda b: record_response("2"))

    return results_df, results_filename    

def reformat_muller_lyer_constantstim_results(results_filename):
    """Reformat the results to indicate if the observer chose the comparison or standard stimulus.
    Currently this is specific to the Muller Lyer illusion"""
    
    # Load results from CSV
    results = pd.read_csv(results_filename, header=0) #, dtype={'trial': np.int32, 'choice': np.int32})
    if results.empty:
        print("Empty file, no data to analyze.")
        return None, None
    
    # create a chooseTop column to be True for choosing the top stimulus as longer, else False
    results['chooseTop'] = np.where(results['response']==1, True, False)
    
    # create a comparison column which is whatever size is *not* the standard 
    # the way PyIllusion works is that the comparison is always bigger than the standard & has the arrows pointing inwards (decreasing its apparent size)
    results['comparison'] = np.where(results['size1'] == results['standard'][0], results['size2'], results['size1'])
    
    # create a delta column which is the absolute difference (the difference column is Pyllusion's difference ratio)
    results['delta'] = results.comparison - results.standard
    
    # create a chooseComparison column to be True for choosing the comparison, else False
    # chooseComparison is True if only ONE of standard1 and chooseTop is True, otherwise False
    results['chooseComparison'] =  np.logical_xor(results['standard1'], results['chooseTop'])
    return results

def get_PSE_JND_constantstim_plot(results, results_fig_filename):
    """Plot the results of the constant stimuli experiment, calculating the PSE and JND"""
    
    # Calculate the proportion of comparison choices for each delta & plot
    props = results.groupby('delta')['chooseComparison'].mean()
    plt.scatter(props.index.tolist(), props.values.tolist(), label="Data")
    plt.xlim(0.0, results['delta'].max()+0.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("How much longer comparison really is")
    plt.ylabel("Fraction of trials the comparison is perceived as longer")
    plt.title("Method of Constant Stimuli")
    
    # Interpolate psychometric function & estimate PSE and JND
    fine = np.linspace(results['delta'].min(), results['delta'].max(), 200)
    interp_props = np.interp(fine, props.index.tolist(), props.values.tolist())
    plt.plot(fine, interp_props, '-', label="Interpolation")
    # Estimate PSE as how much longer the comparison needs to be to be perceived as
    # the same length as the standard on average (prop=0.5)
    PSE = np.interp(0.5, interp_props, fine) 
    # Estimate JND as the mean difference between the 25% and 75% points
    d25 = np.interp(0.25, interp_props, fine) 
    d75 = np.interp(0.75, interp_props, fine)
    JND = (d75 - d25) / 2 # Note that the JND above the PSE and below the PSE may be different, but for simplicity we average them. 
    
    # Plot the PSE and JND lines
    plt.axhline(0.5, color='gray', linestyle='--', label="PSE: Delta = {:.2f}".format(PSE))
    plt.axvline(PSE, color='gray', linestyle='--')
    plt.axhline(0.25, color='gray', linestyle=':', label="PSE - JND, Delta ~= {:.2f}".format(-JND))
    plt.axvline(d25, color='gray', linestyle=':')
    plt.axhline(0.75, color='gray', linestyle=':', label="PSE + JND, Delta ~= {:.2f}".format(JND))
    plt.axvline(d75, color='gray', linestyle=':')
    
    print(f"PSE (50% point): {PSE:.2f}")
    print(f"JND (half 25–75 spread): {JND:.2f}") 
    plt.legend()
    plt.tight_layout()

    # save figure
    plt.savefig(results_fig_filename) 
    print("Saved figure to " + results_fig_filename)
    plt.show()
    return PSE, JND
    

def plot_PSE_stimulus(illusion_type, delta_PSE, illusion_strength_PSE=30, size_min_PSE=0.5):
    """Plot the illusion as it appears at the Point of Subjective Equality (PSE)"""
    plt.figure()
    stimulus = MullerLyer(illusion_strength=illusion_strength_PSE, size_min=size_min_PSE, difference=delta_PSE)
    plt.image(stimulus.to_image())
    plt.title("Illusion at PSE")
    print("The comparison size is " + str(delta_PSE) + " different than the standard size here.")
    print("However the observer only perceives the comparison as longer than the standard 50% of the time.")
    