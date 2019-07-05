from lxml import etree
import numpy as np

class Robot(object):
    """
    An example robot object. Please override me!
    """
    def __init__(self):
        self.morphology = np.ones(shape=(4,4,4))

    def write_to_xml(self, root, **kwargs):
        structure = etree.SubElement(root, "Structure", Compression="ASCII_CSV")
        etree.SubElement(structure, "X_Voxels").text = str(self.morphology.shape[0])
        etree.SubElement(structure, "Y_Voxels").text = str(self.morphology.shape[1])
        etree.SubElement(structure, "Z_Voxels").text = str(self.morphology.shape[2])
        etree.SubElement(structure, "numRegenerationModelSynapses").text = "0"
        etree.SubElement(structure, "numForwardModelSynapses").text = "0"
        etree.SubElement(structure, "numControllerSynapses").text = "0"

        data = etree.SubElement(structure, "Data")
        for layer in range(self.morphology.shape[2]):
            etree.SubElement(data, "Layer").text = etree.CDATA(', '.join([str(d) for d in self.morphology[:,:,layer].flatten()]))
