#!/usr/bin/env python3
"""Simple perceptual experiment using Pygame"""

# import necessary libraries
import pygame
import random
import time
import pyllusion
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

# Global variable configuration (could put these in a config file)
SCREEN_W, SCREEN_H = 800, 600
N_TRIALS_PER_LEVEL = 3  # e.g. 2 trials at each delta
FIXATION_DURATION = 0.5  # seconds
BKGD_COLOR = (255, 255, 255)
CROSS_COLOR =  (0, 0, 0) 
output_data_path = '../expt_results/'

# parameters specific to Muller-Lyer Experiment123
STIMULUS_TYPE = 'MullerLyer'  # or 'image', 'Ebbinghaus', 'Ponzo', 'Zollner', 'Hering', 'Fraser', 'Munker', 'Kanizsa
STIMULUS_DURATION = 0.75  # seconds
INSTRUCTIONS = "Press UP if you see the top line as longer, and DOWN if you see the bottom line as longer"
ILLUSION_STRENGTH = 30
STANDARD_SIZE = 0.5
DELTAS = np.linspace(-0.75, 0.75, 11) #[-0.75, -0.6, -0.45, -0.3, -0.15, 0.15, 0.3, 0.45, 0.6, 0.75]  #DELTA = 1 means top stimulus is 2x larger than bottom stimulus
# Note PyIllusion seems to always point the arrows out on the longer stimulus, which is actually a clue to the observer that I would randomize in a real experiment

def draw_fixation_cross():
    x1 = 1
    x2 = 10
    plt.plot([-x1,x1],[0,0],'k-')
    plt.plot([0,0],[-x1,x1],'k-')
    plt.axis('equal')
    plt.xlim([-x2,x2])
    plt.ylim([-x2,x2])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('off')
    plt.show()
    
def pil_to_pygame(pil_img):
    """Convert a PIL Image to a pygame Surface."""
    mode = pil_img.mode
    size = pil_img.size
    data = pil_img.tobytes()

    # For most modes (RGB, RGBA), pygame.image.fromstring works
    return pygame.image.fromstring(data, size, mode)

def get_stimulus(stimulus_params):
    """Load or generate the stimulus, convert it to a pygame Surface, return any parameters needed for the illusion"""
    # Get the stimulus type and parameters
    stimulus_type = stimulus_params[0]
    if stimulus_type == 'image':
        return get_stimulus_from_image(stimulus_params[1])
    else:
        return get_pyIllusion(stimulus_params)
    
def get_stimulus_from_image(stimulus_path="path/to/your/image.png"):
    """Load a stimulus from an image file and convert it to a pygame Surface"""

    # example of loading your image from a file:
    pil_img = Image.open(stimulus_path).convert("RGBA")

    # Convert to pygame Surface
    return pil_to_pygame(pil_img), []

def init_results_dataframe(stimulus_type):
    """Generate custom results dataframe for a given experiment type"""
    
    column_types = {
        'trial':    pd.Series(dtype='int'),
        'choice':   pd.Series(dtype='int'),
        'RT':       pd.Series(dtype='float')
    }
    if stimulus_type == 'image':
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    elif stimulus_type == 'MullerLyer':
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float'),
                              'standard1': pd.Series(dtype='bool') #standard1 is True if the upper stimulus is the standard (arrows pointing outwards)
        })  
    elif stimulus_type == 'Ebbinghaus':
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    elif stimulus_type == 'Ponzo': 
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    elif stimulus_type == 'Zollner':
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    elif stimulus_type == 'Hering':    
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    elif stimulus_type == 'Fraser':
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    elif stimulus_type == 'Munker':
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    elif stimulus_type == 'Kanizsa':       
        column_types.update({'size1': pd.Series(dtype='float'),
                              'size2': pd.Series(dtype='float')})
    else:
        raise ValueError("init_results_dataframe(): Unknown stimulus type")
    results = pd.DataFrame(column_types) # Create a DataFrame to store results
    return results

def get_pyIllusion(stimulus_params):
    """Generate a stimulus using pyllusion and convert it to a pygame Surface"""

    if stimulus_params[0] == 'MullerLyer':
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.MullerLyer(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)      
    elif stimulus_params[0] == 'Ebbinghaus':
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.Ebbinghaus(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)
    elif stimulus_params[0] == 'Ponzo': 
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.Ponzo(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)
    elif stimulus_params[0] == 'Zollner':
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.Zollner(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)
    elif stimulus_params[0] == 'Hering':    
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.Hering(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)
    elif stimulus_params[0] == 'Fraser':
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.Fraser(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)
    elif stimulus_params[0] == 'Munker':
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.Munker(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)
    elif stimulus_params[0] == 'Kanizsa':       
        (_, ILLUSION_STRENGTH, STANDARD_SIZE, delta) = stimulus_params
        pyIllusion_output = pyllusion.Kanizsa(illusion_strength=ILLUSION_STRENGTH, size_min=STANDARD_SIZE, \
                                    difference=delta)
    else:
        raise ValueError("Unknown stimulus type")

    return pil_to_pygame(pyIllusion_output.to_image()), pyIllusion_output.get_parameters() 

def show_fixation_cross(screen, clock, FIXATION_DURATION, SCREEN_W, SCREEN_H):
    """Display a fixation cross for a specified duration."""
    start_time = time.time()
    while time.time() - start_time < FIXATION_DURATION:
        screen.fill(BKGD_COLOR)
        pygame.draw.line(screen, CROSS_COLOR, (SCREEN_W//2, SCREEN_H//2 - 10), (SCREEN_W//2, SCREEN_H//2 + 10), 2)
        pygame.draw.line(screen, CROSS_COLOR, (SCREEN_W//2 - 10, SCREEN_H//2), (SCREEN_W//2 + 10, SCREEN_H//2), 2)
        pygame.display.flip()
        clock.tick(60)

def run_experiment():
    """Run the experiment"""
    # initialize the window, clock, results
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption(STIMULUS_TYPE)
    clock = pygame.time.Clock()
    results = init_results_dataframe(STIMULUS_TYPE) 
    trials = [delta for delta in DELTAS for _ in range(N_TRIALS_PER_LEVEL)] # Build the trial list
    #random.seed(42) # use a fixed seed if you want reproducibility
    random.shuffle(trials)

    # Main experiment loop, go through each trial
    for trial_index, delta in enumerate(trials):
        # 1) Show pre-trial fixation cross & start reaction timer (RT)
        show_fixation_cross(screen, clock, FIXATION_DURATION, SCREEN_W, SCREEN_H)
        pygame.event.clear()
        responded = False

        # 2) Get the stimulus 
        stimulus_params =  (STIMULUS_TYPE, ILLUSION_STRENGTH, STANDARD_SIZE, delta)
        stimulus, stimulus_output_params = get_stimulus(stimulus_params)
        stimulus_start_time = time.time()

        while not responded:
            # 2) show the stimulus
            screen.fill(BKGD_COLOR)
            screen.blit(stimulus, (0, 0))
            pygame.display.flip()

            # 3) Remove the stimulus and wait for response if the STIMULUS_DURATION time has passed 
            if time.time() - stimulus_start_time > STIMULUS_DURATION:
        
                # clear screen
                screen.fill(BKGD_COLOR)
                show_fixation_cross(screen, clock, FIXATION_DURATION, SCREEN_W, SCREEN_H)
                pygame.display.flip()

                # Wait for and respond to keyboard input
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or event.type == pygame.K_ESCAPE:
                        pygame.quit()
                        return results
                    elif event.type == pygame.KEYDOWN and event.key in (pygame.K_UP, pygame.K_DOWN):
                        rt = time.time() - stimulus_start_time
                        choice = 0 if event.key == pygame.K_UP else 1
                        standard1 = stimulus_output_params['Distractor_TopLeft1_x1'] > stimulus_output_params['Distractor_TopLeft1_x2']
                        results.loc[len(results)] = [trial_index, choice, rt, \
                                                    stimulus_output_params['Size_Top'], 
                                                    stimulus_output_params['Size_Bottom'],
                                                    standard1]

                        responded = True
                        break

            #clock.tick(60)

    pygame.quit()
    return results

def save_results_to_csv(results, results_filename="results.csv"):
    results.to_csv(results_filename, index=False, header=True)
    print(f"Saved {len(results)} rows to {results_filename}")

def analyze_results(results_filename):
    """Analyze the results and print out the PSE and JND"""

    # Load results from CSV
    results = pd.read_csv(results_filename, header=0, dtype={'trial': np.int32, 'choice': np.int32})
    if results.empty:
        print("Empty file, no data to analyze.")
        return None, None

    # determine standard size and put it in a new column
    # although I might design it differently in the future, the way PyIllusion works is that the standard size 
    # is always smaller than the comparison size and always has the arrows pointing outwards (increasing its apparent size)
    possible_sizes = np.unique(results[['size1', 'size2']])
    for s in possible_sizes:
        if results[['size1', 'size2']].isin([s]).any(axis=1).all():
            estimated_standard_size = s
            print("Standard size: ", estimated_standard_size)
            break
    results['standard'] = estimated_standard_size # create a standard size column (constant)

    # create a comparison column which is whatever size is *not* the standard 
    # although I might design it differently in the future, the way PyIllusion works is that the comparison size 
    # is always bigger than the standard size and always has the arrows pointing inwards (decreasing its apparent size)
    results['comparison'] = np.where(
        results['size1'] == estimated_standard_size,
        results['size2'],
        results['size1']
    )
    results['delta'] = results.comparison - results.standard

    # create a chooseComparison column to be 0 for choosing the standard stimulus and 1 for comparison stimulus
    # if the top stimulus is the standard, then chooseComparison is the same as choice
    # if the top stimulus is the comparison, then chooseComparison is the opposite of choice
    results['chooseComparison'] = np.where(results['standard1'], results['choice'], abs(results['choice'] - 1))
    
    # Calculate the proportion of comparison choices for each delta & plot
    props = results.groupby('delta')['chooseComparison'].mean()
    plt.scatter(props.index.tolist(), props.values.tolist(), label="Data")
    plt.xlim(0.0, results['delta'].max()+0.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("How much longer comparison really is")
    plt.ylabel("Fraction of trials the comparison is perceived as longer")
    plt.title("Muller-Lyer Illusion")

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
    JND = (d75 - d25) / 2 

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
    plt.show()

    return PSE, JND

def main():

    # set up results file name
    observer_ID = input("Enter observer ID: ")
    datetime_string = datetime.now().strftime("%Y-%m-%d-%H-%M")
    if os.path.exists(output_data_path) == False:
        os.makedirs(output_data_path)
    results_filename = output_data_path + STIMULUS_TYPE + "_" + str(observer_ID) + "_" + datetime_string + '_results.csv'

    # Print instructions
    print(INSTRUCTIONS)
    print("Press ESC to quit")
    input("Press Enter to start the experiment")

    # run experiment
    results = run_experiment()

    # Save results to a CSV file 
    save_results_to_csv(results, results_filename=results_filename)

    # analyze results from file
    PSE, JND = analyze_results(results_filename)

if __name__ == "__main__":
    main()
