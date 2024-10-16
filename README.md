The codes are adapted from: 1) ITI-GEN: https://github.com/humansensinglab/ITI-GEN; 2) DAAM: https://github.com/castorini/daam

## FairQueue
- ./ckpts contain the saved ITI-GEN embeddings listed in the main paper. Note that the "basis_final_embed_29.pt" are the final embeddings utilized in sample generation (this is the same as the orignal base code).

1) Run generation_Mixing.py file with the command:
```
Python generation_Mixing.py --attr-list <tSA> --prompt-path <Location of Prompt path ending with 'basis_final_embed_29.pt'>
```

Please find the ./models , ./iti_gen and ./ckpts files at [link](https://drive.google.com/drive/folders/1iBizW8YhmvDUo6ChTIhN9f_M1EnIH-hs)

## Cross-attention Analysis
- Put the folder './ckpts' under the project folder
- Establish the environment following either requirement.txt or environment.yml (similar to daam 0.1.0 in https://github.com/castorini/daam)
- Replace the original file 'pipeline_stable_diffusion.py' under the environment '/lib/python3.8/site-packages/diffusers/pipelines/stable_diffusion' with the one provided in the folder
- One may run 'cros_attn_analysis.py' with the command:
```
./run_cros_attn_analysis.sh --tSA "Smiling" --prompts_mode "H2I" --sample_list "[1,2]" --step_start_switch 15
```
One may find 'cros_attn_analysis.py' for other parameters that can be assigned

