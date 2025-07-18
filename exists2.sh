#!/bin/bash

# Script to check if specific URLs exist in a text file
# Usage: ./check_urls_in_file.sh <text_file.txt>

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# URLs to check for
URLS_TO_CHECK=(
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
    "https://github.com/lamberev/tc_benchmark/tree/main"
    "https://github.com/larkinandy/Matching_HEI_4970"
    "https://github.com/LeoLjl/Fine_grained_optimize"
    "https://github.com/LucasKolanz/DECCO"
    "https://github.com/OSU-IDEA-Lab/scalable-data-integration-with-LLMs"
    "https://github.com/pcarlip/multigpu-ocean"
    "https://github.com/petejacobs/Huang-Complex-Supercomputer-seed-grant/"
    "https://github.com/SamCT/HuangSuperCompute_Proposal"
    "https://github.com/WooJin-Cho/Parameterized-Physics-informed-Neural-Networks/"
)

# Function to display usage
usage() {
    echo "Usage: $0 <text_file.txt>"
    echo "Example: $0 document.txt"
    exit 1
}

# Check if file argument is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No file provided${NC}"
    usage
fi

# Check if file exists
if [ ! -f "$1" ]; then
    echo -e "${RED}Error: File '$1' not found${NC}"
    exit 1
fi

TEXT_FILE="$1"
TOTAL_URLS=${#URLS_TO_CHECK[@]}
FOUND_URLS=0
MISSING_URLS=0

echo -e "${YELLOW}Checking for URLs in: $TEXT_FILE${NC}"
echo "=================================="

# Check each URL in the text file
for url in "${URLS_TO_CHECK[@]}"; do
    echo -n "Checking: $url ... "
    
    # Use grep to search for the URL in the file
    # -F: treat pattern as fixed string (not regex)
    # -q: quiet mode (no output, just exit status)
    if grep -Fq "$url" "$TEXT_FILE"; then
        echo -e "${GREEN}✓ FOUND${NC}"
        FOUND_URLS=$((FOUND_URLS + 1))
    else
        echo -e "${RED}✗ NOT FOUND${NC}"
        MISSING_URLS=$((MISSING_URLS + 1))
    fi
done

echo ""
echo "=================================="
echo -e "${YELLOW}Summary:${NC}"
echo "Total URLs checked: $TOTAL_URLS"
echo -e "Found in file: ${GREEN}$FOUND_URLS${NC}"
echo -e "Missing from file: ${RED}$MISSING_URLS${NC}"

if [ $MISSING_URLS -eq 0 ]; then
    echo -e "${GREEN}All URLs found in the file!${NC}"
    exit 0
else
    echo -e "${RED}Some URLs are missing from the file.${NC}"
    exit 1
fi
