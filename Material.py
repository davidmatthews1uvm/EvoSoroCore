import random
from lxml import etree


class Material(object):
    """
    base class for materials.
    Please override if you need special attributes such as actuation or cilia
    """
    def __init__(self, id, name="", elastic_mod=0, friction_static=0, friction_dynamic=0, density=0, color=(1,1,0,1), actuation_variance=0):
        self.id = id
        self.name = name
        self.elastic_mod = elastic_mod
        self.friction_static = friction_static
        self.friction_dynamic = friction_dynamic
        self.density = density
        self.color = color
    def write_to_xml(self, root, **kwargs):
        material_root = etree.SubElement(root, "Material", ID=str(self.id))
        etree.SubElement(material_root, "MatType").text = str(0)
        etree.SubElement(material_root, "Name").text = str(self.name)

        display = etree.SubElement(material_root, "Display")
        etree.SubElement(display, "Red").text = str(self.color[0])
        etree.SubElement(display, "Green").text = str(self.color[1])
        etree.SubElement(display, "Blue").text = str(self.color[2])
        etree.SubElement(display, "Alpha").text = str(self.color[3])

        mechanical = etree.SubElement(material_root, "Mechanical")
        etree.SubElement(mechanical, "MatModel").text = "0"
        etree.SubElement(mechanical, "Elastic_Mod").text = str(self.elastic_mod)
        etree.SubElement(mechanical, "Plastic_Mod").text = "0"
        etree.SubElement(mechanical, "Yield_Stress").text = "0"
        etree.SubElement(mechanical, "FailModel").text = "0"
        etree.SubElement(mechanical, "Fail_Stress").text = "0"
        etree.SubElement(mechanical, "Fail_Strain").text = "0"
        etree.SubElement(mechanical, "Density").text = str(self.density)
        etree.SubElement(mechanical, "Poissons_Ratio").text = "0.35"
        etree.SubElement(mechanical, "CTE").text = "0"
        etree.SubElement(mechanical, "uStatic").text = str(self.friction_static)
        etree.SubElement(mechanical, "uDynamic").text = str(self.friction_dynamic)