#!/bin/bash

# --- Script to check the existence of GitHub repositories ---

# Array of GitHub repository URLs to check.
# The list has been cleaned from the image (removed query strings, etc.).
repos=(
    "https://github.com/ag2ai/simpledoc"
    "https://github.com/AI4FoodUniverse/FoodImage"
    "https://github.com/ai2cm/ace"
    "https://github.com/apple/ml-autofocusformer"
    "https://github.com/BenjaminDalziel/ampnet"
    "https://github.com/ccsb-scripps/AutoDock-GPU"
    "https://github.com/CEMeNT-PSAAP/MCDC"
    "https://github.com/CliMA/Oceananigans.jl"
    "https://github.com/elsaessern/Bethe-Functions"
    "https://github.com/eMapR/sundial"
    "https://github.com/haydenjohnson94/HuangSeedFund_Evo2"
    "https://github.com/hazgrav/discovery"
    "https://github.com/HumainLab/Understand_MarginPO"
    "https://github.com/karkisa/special-train"
    "https://github.com/lamberev/tc_benchmark"
    "https://github.com/larkinandy/Matching_HEI_4970"
    "https://github.com/LeoLjl/Fine_grained_optimize"
    "https://github.com/LucasKolanz/DECCO"
    "https://github.com/OSU-IDEA-Lab/scalable-data-integration-with-LLMs"
    "https://github.com/pcarlip/multigpu-ocean"
    "https://github.com/petejacobs/Huang-Complex-Supercomputer-seed-grant"
    "https://github.com/SamCT/HuangSuperCompute_Proposal"
    "https://github.com/WooJin-Cho/Parameterized-Physics-informed-Neural-Networks"
)

# Array to store non-existent repos
not_found=()

echo "--- Checking GitHub Repositories ---"

# Loop through the array and check each repo
for url in "${repos[@]}"; do
    # Get the HTTP status code
    # -s: silent mode
    # --head: fetch headers only (faster)
    # -L: follow redirects (important for GitHub)
    # -w "%{http_code}": write out the status code
    # -o /dev/null: discard the body of the response
    status_code=$(curl -s --head -L -w "%{http_code}" "$url" -o /dev/null)

    if [ "$status_code" -eq 200 ]; then
        echo "[  OK  ] $url"
    else
        # A 404 status means "Not Found"
        echo "[ FAIL ] $url (Status: $status_code)"
        not_found+=("$url")
    fi
done

echo ""
echo "--- Summary ---"

# Check if the not_found array is empty
if [ ${#not_found[@]} -eq 0 ]; then
    echo "All repositories were found!"
else
    echo "The following repositories returned a 404 'Not Found' error:"
    # Print each non-existent URL
    for url in "${not_found[@]}"; do
        echo " - $url"
    done
fi
