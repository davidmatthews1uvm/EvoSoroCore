from lxml import etree
from Simulator import Sim
from Environment import Env, VXC_Wrapper
from Material import Material
from Robot import Robot

class Individual(object):
    def __init__(self):
        pass

    def get_id(self):
        return 0

    def get_body_xlim(self):
        return 1

    def get_body_ylim(self):
        return 1

    def get_body_zlim(self):
        return 1


root = etree.Element("VXA", Version="1.0")
Sim().write_to_xml(root, "TestDir", Individual())
Env().write_to_xml(root, "TestDir", Individual())
VXC_Wrapper().write_to_xml(root, [Material(1, name="Test1"), Material(2, name="Test2")], Robot())


xml_str = etree.tostring(root, pretty_print=True, xml_declaration=True, encoding="ISO-8859-1").decode("utf-8")

with open("to_sim.vxa", "w") as f:
    f.write(xml_str)
print(xml_str)

