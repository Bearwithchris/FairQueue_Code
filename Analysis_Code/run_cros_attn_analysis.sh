#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 [--tSA <tSA>] [--device <device>] [--prompts_mode <prompts_mode>] [--path <path>] [--sample_list <sample_list>] [--step_start_switch <step_start_switch>]"
    echo "  --tSA <tSA>                Target sensitive attribute (e.g., 'Smiling')"
    echo "  --device <device>          Device to use (e.g., 'cuda:0')"
    echo "  --prompts_mode <prompts_mode>   Prompts Mode (e.g., 'ITI_2_HP')"
    echo "  --path <path>              Project Path (e.g., '/home/xxxx/Projects/FairQueue')"
    echo "  --sample_list <sample_list>      List of sample indexes (e.g., '[0, 1, 2, 3, 4, 5]')"
    echo "  --step_start_switch <step_start_switch> Step Start switch (e.g., 15)"
    exit 1
}

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --tSA) tSA="$2"; shift 2 ;;
        --device) device="$2"; shift 2 ;;
        --prompts_mode) prompts_mode="$2"; shift 2 ;;
        --path) path="$2"; shift 2 ;;
        --sample_list) sample_list="$2"; shift 2 ;;
        --step_start_switch) step_start_switch="$2"; shift 2 ;;
        *) echo "Unknown parameter passed: $1"; usage ;;
    esac
done

# Prepare the command with optional parameters
cmd="python cros_attn_analysis.py"
[ -n "$tSA" ] && cmd="$cmd --tSA \"$tSA\""
[ -n "$device" ] && cmd="$cmd --device \"$device\""
[ -n "$prompts_mode" ] && cmd="$cmd --prompts_mode \"$prompts_mode\""
[ -n "$path" ] && cmd="$cmd --path \"$path\""
[ -n "$sample_list" ] && cmd="$cmd --sample_list \"$sample_list\""
[ -n "$step_start_switch" ] && cmd="$cmd --step_start_switch $step_start_switch"

# Execute the command
eval $cmd
