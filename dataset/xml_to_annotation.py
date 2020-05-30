import xml.etree.ElementTree as ET
import os

DATADIR = "data"

file = open("data.txt","r+")

path = os.path.join(DATADIR)
for xml in os.listdir(path):
    tree = ET.parse(os.path.join(DATADIR, xml))
    root = tree.getroot()
    #file name
    name = root[1].text
    xmin = root[6][4][0].text
    xmax = root[6][4][3].text
    ymin = root[6][4][1].text
    ymax = root[6][4][3].text
    cat = root[6][0].text
    file.write("\ndataset/destimges/" + name + "," + xmin + "," + ymin + "," + xmax + "," + ymax + "," + cat)

file.close()
