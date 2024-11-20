rem Step 1: Activate Anaconda
call "%userprofile%\anaconda3\Scripts\activate.bat" "%userprofile%\anaconda3"

rem Step 2: remove the environment
call conda remove --name searcher --all -y

cmd /k