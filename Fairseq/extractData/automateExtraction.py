import subprocess
message = "The exit code was: "
scrapped_text = subprocess.run(["python", "extractData/webScrap.py", "https://www.bod.asia/category/flash-news/", "720", "730", "extractData/output.txt"])
print("%s: %d" % (message, scrapped_text.returncode))
sentences_text = subprocess.run(["python", "extractData/breakIntoSentence.py", "extractData/output.txt", "data/Nov1TibetNet.txt"])
print("%s: %d" % (message, sentences_text.returncode))
# list_files = subprocess.run(["python", "cleanData/cleanRawTibData.py", "broken.txt", "clean.txt"])
# print("The exit code was: %d" % list_files.returncode)
