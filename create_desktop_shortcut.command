#!/bin/bash
# create_desktop_shortcut.command
# 
# This script creates desktop shortcuts that properly point to the project directory

# Get the directory where this script is located (project directory)
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create the test launcher
cat > ~/Desktop/Style_Comparison.command << EOL
#!/bin/bash
cd "${PROJECT_DIR}"
source ./venv/bin/activate
echo "==============================================================="
echo "          MULTI-STYLE CONTENT GENERATOR TEST                   "
echo "==============================================================="
echo ""
echo "This application will show content generation in two styles:"
echo "1. C. Pete Connor - Data-driven tech expert style"
echo "2. The Onion - Satirical news style"
echo ""
echo "Starting the test now..."
echo ""
python3 "${PROJECT_DIR}/test_multi_style.py"
echo ""
echo "==============================================================="
echo "Test completed. Press Enter to close this window."
echo "==============================================================="
read
EOL

# Create the Pete Connor app launcher
cat > ~/Desktop/PeteConnor_App.command << EOL
#!/bin/bash
cd "${PROJECT_DIR}"
source ./venv/bin/activate
export DEFAULT_STYLE="pete_connor"
echo "==============================================================="
echo "          PETE CONNOR STYLE CONTENT GENERATOR                  "
echo "==============================================================="
echo ""
echo "This application allows you to generate satirical content"
echo "in the style of C. Pete Connor across multiple platforms."
echo ""
echo "This launcher runs the app with 'pete_connor' style preset."
echo ""
echo "Starting the app now..."
echo ""
python3 "${PROJECT_DIR}/app.py"
if [ \$? -ne 0 ]; then
    echo ""
    echo "==============================================================="
    echo "App exited with an error. Press Enter to close this window."
    echo "==============================================================="
    read
fi
EOL

# Create the Onion app launcher
cat > ~/Desktop/TheOnion_App.command << EOL
#!/bin/bash
cd "${PROJECT_DIR}"
source ./venv/bin/activate
export DEFAULT_STYLE="onion"
echo "==============================================================="
echo "          THE ONION STYLE CONTENT GENERATOR                    "
echo "==============================================================="
echo ""
echo "This application allows you to generate satirical content"
echo "in the style of The Onion across multiple platforms."
echo ""
echo "This launcher runs the app with 'onion' style preset."
echo ""
echo "Starting the app now..."
echo ""
python3 "${PROJECT_DIR}/app.py"
if [ \$? -ne 0 ]; then
    echo ""
    echo "==============================================================="
    echo "App exited with an error. Press Enter to close this window."
    echo "==============================================================="
    read
fi
EOL

# Make the new shortcuts executable
chmod +x ~/Desktop/Style_Comparison.command ~/Desktop/PeteConnor_App.command ~/Desktop/TheOnion_App.command

echo "Desktop shortcuts created successfully!"
echo "You can now run:"
echo "1. Style_Comparison.command - Shows content generation in both styles"
echo "2. PeteConnor_App.command - Launches the app with Pete Connor style preset"
echo "3. TheOnion_App.command - Launches the app with Onion style preset"
echo ""
echo "Press Enter to close this window."
read