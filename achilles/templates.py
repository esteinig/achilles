from textwrap import dedent
from colorama import Fore

C = Fore.CYAN
R = Fore.RED
G = Fore.GREEN
LB = Fore.LIGHTBLUE_EX
RE = Fore.RESET
M = Fore.MAGENTA
Y = Fore.YELLOW


def get_inspect_model_message():
    return f"Inspect models with: {M}achilles inspect -m {Y}<model>{RE}"


def get_inspect_oollection_message():
    return f"Inspect collections with: {M}achilles inspect -c {Y}<collection>{RE}"


def get_param_template(ds, tr):

    return dedent(
        f"""
        {C}Dataset{R} 
        {RE}=========={RE} 

        {M}Global tags{RE}              {G}{', '.join(ds['global_tags'])}{RE} 
        {M}Sample files per tag{RE}     {G}{ds['sample_files_per_tag']}{RE} 
        {M}Proportional sampling{RE}    {G}{ds['sample_proportions']}{RE} 
        
        {C}Signal Sampling{RE} 
        {RE}================{RE} 
        
        {M}Windows per label{RE}        {G}{ds['max_windows']}{RE} 
        {M}Windows per read{RE}         {G}{ds['max_windows_per_read']}{RE} 
        {M}Window size{RE}              {G}{ds['window_size']}{RE} 
        {M}Window step{RE}              {G}{ds['window_step']}{RE} 
        {M}Window recovery{RE}          {G}{ds['window_recover']}{RE} 
        {M}Random start{RE}             {G}{ds['window_random']}{RE}
        
        {C}Architecture{RE} 
        {RE}============={RE} 

        {M}Size of input layer{RE}      {G}{tr['window_size']}{RE} 
        {M}Residual blocks{RE}          {G}{tr['nb_residual_block']}{RE}    
        {M}LSTM layers{RE}              {G}{tr['nb_rnn']}{RE} 
        {M}Output activation{RE}        {G}{tr['activation']}{RE} 
        
        {C}Training{RE} 
        {RE}========={RE} 

        {M}Epochs{RE}                   {G}{tr['epochs']}{RE} 
        {M}Batch size{RE}               {G}{tr['batch_size']}{RE} 
        {M}Optimizer{RE}                {G}{tr['optimizer']}{RE} 
        {M}Loss function{RE}            {G}{tr['loss']}{RE} 
        
        {C}Regularization{RE} 
        {RE}==============={RE} 

        {M}Dropout{RE}                  {G}{tr['dropout']}{RE} 
        {M}Recurrent dropout{RE}        {G}{tr['recurrent_dropout']}{RE}
        
        """
    )