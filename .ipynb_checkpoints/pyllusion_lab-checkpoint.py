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
def save_results_to_csv(results, results_fname="../expt_results/results.csv"):
    """save the results dataframe to results_fname"""
    results.to_csv(results_fname, index=False, header=True)
    print(f"Saved {len(results)} rows to {results_fname}")


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


def render_illusion(illusion_type, illusion_strength, standard, difference=0):
    """Render a pyllusion illusion, return it as a Pillow image and its parameters"""
    illusion = eval(f"pyllusion.{illusion_type}(illusion_strength={illusion_strength}, size_min={standard}, difference={difference})")
    return illusion.to_image(), illusion.get_parameters() 


def pre_render_stimuli(illusion_type, illusion_strengths, standard, differences, 
                       prerendered_stimuli_dir="../expt_results/pre_rendered_stimuli/"):
    """
    Pre-render and save stimuli for a set of illusion_strengths and set of delta values.
pyllusion_adjustment_expt
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
    stimuli_df : pd.DataFrame
    """
    os.makedirs(prerendered_stimuli_dir, exist_ok=True) # Create output directory if it doesn't exist
    stimulus_meta_fname = os.path.join(prerendered_stimuli_dir, f"{illusion_type}.csv")

    # check to see if we've already pre-rendered all these stimuli
    if os.path.exists(stimulus_meta_fname):
        stimuli_df = pd.read_csv(stimulus_meta_fname)
        if len(stimuli_df) >= len(illusion_strengths)*len(differences):
            if np.all([s in stimuli_df["illusion_strength"].unique() for s in illusion_strengths]) and \
            np.all([s in stimuli_df["difference"].unique() for s in differences]) and \
            np.all([s in stimuli_df["standard"].unique() for s in [standard]]):
                print(f"{len(stimuli_df)} of the right stimuli already exist, not rendering more")
                return stimuli_df
        
    
    # if not, render each
    stimuli_df = pd.DataFrame(columns=['illusion_type', 'illusion_strength', 'standard', 'difference', 'size1', 'size2', 'standard1', 'image_path'])
    for strength in illusion_strengths:
        for delta in differences:
            print(strength, delta)
            # Render the illusion once
            stimulus, stimulus_params = render_illusion(illusion_type, strength, standard, delta)
            
            # Save the image
            fname = f"{illusion_type}_strength{strength}_std{standard}_delta{delta}.png"
            filepath = os.path.join(prerendered_stimuli_dir, fname)
            stimulus.save(filepath)
            
            # code up some of the particulars of pyllusion to save
            if illusion_type == 'Ebbinghaus':
                size1 = stimulus_params["Size_Inner_Left"]
                size2 = stimulus_params["Size_Inner_Right"]
                standard1 = True if stimulus_params['Size_Inner_Left'] == standard else False
            elif illusion_type == 'MullerLyer':                    
                size1 = stimulus_params["Size_Top"]
                size2 = stimulus_params["Size_Bottom"]
                standard1 = True if stimulus_params['Distractor_TopLeft1_x1'] > stimulus_params['Distractor_TopLeft1_x2'] else False
            else:
                print(f"Unknown illusion type {illusion_type}")
                size1 = None
                size2 = None
                standard1 = None                
            # record all the necessary stimulus meta data
            stimuli_df.loc[len(stimuli_df)] = [illusion_type, strength, standard, delta, \
                                               size1, size2, standard1, filepath]

    # save all the meta data to a csv
    stimuli_df.to_csv(stimulus_meta_fname, index=False)
    print(str(len(illusion_strengths)*len(differences)) + " illusion stimuli written to " + prerendered_stimuli_dir)
    print(f"Meta data written to {stimulus_meta_fname}")
    
    return stimuli_df

    
def load_prerendered_stimulus(stimuli_df, illusion_type, illusion_strength=30, difference=0, standard=0.5):
    """Load a pre-rendered stimulus, returns the Pillow image and the pyllusion parameters"""
    stimulus_img, stimulus_params = None, None
    idx = np.where(\
        (stimuli_df.illusion_type == illusion_type) & (stimuli_df.illusion_strength == illusion_strength) & \
        (stimuli_df.standard == standard) & (stimuli_df.difference == difference))[0] 
    if len(idx) == 0:
        print("Can't find image!")
    else:
        image_fname = stimuli_df.iloc[idx[0]]["image_path"]
        if os.path.exists(image_fname):
            stimulus_img = Image.open(image_fname)            
            stimulus_params = stimuli_df.iloc[idx[0]]    
    return stimulus_img, stimulus_params
    

def pyllusion_adjustment_expt(illusion_type, illusion_strength=30, standard=0.5, instructions="", slider_min=0.0, slider_max=1.0, img_scale=0.25):
    """An simple illusion perception experiment using the Method of Adjustment and a pyllusion-generated stimulus.
    The observer adjusts a slider until the two stimuli look the same, and then presses the Submit button"""

    # make sure the illusion_type is available
    if not check_illusion_type(illusion_type):
        return
        
    # Define a frame update "event handler" function that redraws the stimulus image
    def update_image(change):
        delta = change['new'] # this is the slider's value
        with output:
            output.clear_output(wait=True)
            # generate the pyllusion stimulus (change the next line if you want to generate the stimulus a different way)
            img, _ = render_illusion(illusion_type, illusion_strength, standard, delta)
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

def plot_adjustment_results(adjustment_results_df, results_fig_fname):
    """plot summary of your results for across multiple illusion strengths (arrow angles)"""
    adjustment_results_df.plot('illusion_strength','PSE', marker='o', linestyle='-')
    plt.xlabel('Illusion strength')
    plt.ylabel('Extra comparison size needed\n to be perceived as same as standard')
    plt.title('Method of Adjustment')
    plt.tight_layout()
    plt.savefig(results_fig_fname) 
    print("Saved figure to " + results_fig_fname)
    plt.show()

def centered_text(content):
    display(HTML(f"<div style='text-align: center;'>{content}</div>"))

def save_results_to_csv(df, results_fname):
    df.to_csv(results_fname, index=False)
    print(f"Results saved to {results_fname}")

def pyllusion_constantstim_expt(illusion_type, illusion_strength=30, differences=[-1, -0.5, 0, 0.5, 1], num_trials_per_level=2, standard=0.5,
                                duration=0.8, output_data_path="../expt_results/", welcome_instructions="Welcome!", trial_instructions="Which looked bigger?", 
                                stimuli_df={}, img_scale=0.5):
    """An simple illusion perception experiment using the Method of Constanst Stimuli and a pyllusion-generated stimulus.
    The observer clicks one of two buttons on repeated trials in random order"""

    # make sure the illusion_type is available
    if not check_illusion_type(illusion_type):
        return

    # setup UX
    messages = Output(layout=Layout(display='flex', justify_content='center'))
    image_box = Output(layout=Layout(display='flex', justify_content='center')) # a centered box for the illusion stimulus, which starts empty
    controls = VBox(layout=Layout(align_items='center')) 
    main_ui = VBox([image_box, messages, controls], layout=Layout(align_items='center')) # a centered box with the illusion, the messages and controls
    display(main_ui)
    observer_id_input = Text(description="Observer ID:", placeholder="e.g. 001")
    start_button = Button(description="Start Experiment", button_style='success')
    controls.children = [observer_id_input, start_button]
    with messages:
        messages.clear_output(wait=True)
        print(welcome_instructions)    
    try:
        fixation_cross = Image.open("./fixation_cross.jpg")
    except FileNotFoundError:
        with output:
            print("Error: fixation_cross.jpg not found.")
        return

    # set up trials using method of constant stimuli
    deltas = np.random.permutation(differences * num_trials_per_level) # randomized order of stimuli
    results_df = pd.DataFrame(columns=['trial', 'illusion_strength', 'standard', 'difference', 'response', 'RT', 'size1', 'size2', 'standard1', 'chooseComparison'])
    trial_data = {"i": 0, "start_time": None, "results": results_df, "standard1": None, "size1": None, "size2": None}
    results_fname= ""   

    def show_buttons():
        """Show two response buttons"""        
        # 1) Replace stimulus with fixation cross
        with image_box:
            clear_output(wait=True)
            display(pil_to_ipyimage(fixation_cross, scale=img_scale)) 
        
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
        nonlocal results_fname
        # trial number
        i = trial_data["i"]        
        if i >= len(deltas): # end of experiment
            with messages:
                messages.clear_output(wait=True)
                print("Experiment complete. Data saved to " + results_fname)
            controls.children = []
            save_results_to_csv(trial_data["results"], results_fname=results_fname)
            return 
            
        # load the pre-generated stimulus & parameters
        stimulus_img, stimulus_params = load_prerendered_stimulus(stimuli_df, illusion_type, illusion_strength=illusion_strength, difference=deltas[i], standard=standard)
        trial_data["size1"] = stimulus_params['size1']
        trial_data["size2"] = stimulus_params['size2']
        trial_data["standard1"] = stimulus_params['standard1']

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
        if illusion_type == "Ebbinghaus" or illusion_type == "MullerLyer":
            # choseFirst is True if observer chose the left/top stimulus
            chooseFirst = np.where(response_code==1, True, False)
            # chooseComparison is True if only ONE of standard1 and chooseFirst is True, otherwise False
            chooseComparison = np.logical_xor(trial_data["standard1"], chooseFirst)
        else:
            chooseComparison = None
        results_df.loc[len(results_df)] = [i, illusion_strength, standard, deltas[i - 1], response_code, rt, trial_data["size1"], trial_data["size2"], trial_data["standard1"], chooseComparison]                
        trial_data["results"] = results_df

        # Clear and proceed with the next trial
        image_box.clear_output(wait=True)
        messages.clear_output(wait=True)
        controls.children = []
        run_trial()

    def on_start_clicked(b):
        """Call once, when the start button is clicked"""
        nonlocal results_fname
        observer_id = observer_id_input.value.strip()
        with messages:
            if not observer_id:
                messages.clear_output(wait=True)
                print("Please enter an Observer ID before starting.")
                return
            else:
                messages.clear_output(wait=False) # clear the welcome instructions
        controls.children = [] # clear the start button
        results_fname = f"{output_data_path}{illusion_type}_{illusion_strength}_constant_stimuli_{observer_id}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}_results.csv"
        run_trial()


    # button set up and callbacks
    start_button.on_click(on_start_clicked)
    top_button = Button(description="Left / Top Bigger", button_style='info', layout=Layout(width="400px"))
    bottom_button = Button(description="Right / Bottom Bigger", button_style='info', layout=Layout(width="400px"))
    top_button.on_click(lambda b: record_response("1"))
    bottom_button.on_click(lambda b: record_response("2"))

    def get_fname():
        nonlocal results_fname
        return results_fname

    return results_df, get_fname


def get_PSE_JND_constantstim_plot(results_fname):
    """Plot the results of the constant stimuli experiment, calculating the PSE and JND"""
    # load data
    constantstimuli_results_df = pd.read_csv(results_fname)
    results_fig_fname = results_fname.rstrip('_results.csv') + '.jpg'
    
    # Calculate the proportion of comparison choices for each delta & plot
    props = constantstimuli_results_df.groupby('difference')['chooseComparison'].mean()
    plt.scatter(props.index.tolist(), props.values.tolist(), label="Data")
    plt.xlim(constantstimuli_results_df['difference'].min()-0.05, constantstimuli_results_df['difference'].max()+0.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("How much bigger comparison really is")
    plt.ylabel("Fraction of trials the comparison is perceived as bigger")
    plt.title("Method of Constant Stimuli")
    
    # Interpolate psychometric function & estimate PSE and JND
    fine = np.linspace(constantstimuli_results_df['difference'].min(), constantstimuli_results_df['difference'].max(), 200)
    interp_props = np.interp(fine, props.index.tolist(), props.values.tolist())
    plt.plot(fine, interp_props, '-', label="Interpolation")
    # Estimate PSE as how much longer the comparison needs to be to be perceived as
    # the same length as the standard on average (prop=0.5)
    if interp_props[0]<interp_props[-1]:
        PSE = np.interp(0.5, interp_props, fine) 
        d25 = np.interp(0.25, interp_props, fine) 
        d75 = np.interp(0.75, interp_props, fine)
    else: # np.interp only works if the 2nd arg is in ascending order
        PSE = np.interp(0.5, np.flip(interp_props), np.flip(fine)) 
        d25 = np.interp(0.25, np.flip(interp_props), np.flip(fine)) 
        d75 = np.interp(0.75, np.flip(interp_props), np.flip(fine)) 
    JND = abs(d75 - d25) / 2  # Estimate JND as the mean difference between the 25% and 75% points        
    
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
    plt.savefig(results_fig_fname) 
    print("Saved figure to " + results_fig_fname)
    plt.show()
    return PSE, JND
    
def plot_illusion_at_PSE_JND(illusion_type, PSE, JND, illusion_strength=30, standard=0.5):
    """Plot three versions of the illusion: at the PSE, as well as one JND above and below"""
    
    illusion_JND_below_PSE, _ = render_illusion(illusion_type, illusion_strength, standard, PSE-JND)
    illusion_PSE, _ = render_illusion(illusion_type, illusion_strength, standard, PSE)
    illusion_JND_above_PSE, _ = render_illusion(illusion_type, illusion_strength, standard, PSE+JND)

    print(f"{illusion_type} illusion with strength {illusion_strength} at the PSE and one JND below and above. Can you perceive the difference? Probably not easily.")
    
    plt.figure(figsize=(12,8))  
    plt.subplot(1,3,1)
    plt.imshow(illusion_JND_below_PSE)
    plt.axis('off')
    plt.title("PSE - JND")
    
    plt.subplot(1,3,2)
    plt.imshow(illusion_PSE)
    plt.axis('off')
    plt.title("PSE")
    
    plt.subplot(1,3,3)
    plt.imshow(illusion_JND_above_PSE)
    plt.axis('off')
    plt.title("PSE + JND")
    
    plt.show()
    