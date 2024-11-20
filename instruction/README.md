Step 1 to 5 only needs to be done once

Step 1. Install Anaconda
From https://www.anaconda.com/download -> free download -> skip registration

Step 2. Install Git
From https://git-scm.com/downloads

Step 3. Install Microsoft Visaul C++
From https://visualstudio.microsoft.com/visual-cpp-build-tools -> download Build Tools

Step 4. Install Tesseract
From https://github.com/UB-Mannheim/tesseract/wiki -> download latest installer

Step 4.1 Add tesseract to PATH
Go to local disk(C:) -> Program Files -> Tesseract-OCR, and copy the full path.
Click Setting icon in your computer -> search System or About -> Advanced system settings -> Environment variable(N)... -> click path -> click add(N) -> paste the full path that you copied before.

Step 5. Install app
For CPU only, execute install_cpu.bat , for GPU support execute install_gpu.bat( Need to install NVIDIA GPU driver first), for user do not know what's the differences between two of them -> execute install_cpu.bat (Download time might takes 3 to 5 minutes.)

Step 6. Run the app
Execute run.bat (First time run the app might takes 3 to 5 minutes to preprocessing and generate embedding.)
