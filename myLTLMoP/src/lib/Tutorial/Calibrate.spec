# This is a specification definition file for the LTLMoP toolkit.
# Format details are described at the beginning of each section below.


======== SETTINGS ========

Actions: # List of action propositions and their state (enabled = 1, disabled = 0)

CompileOptions:
convexify: False
fastslow: False
symbolic: False
decompose: True
use_region_bit_encoding: True
synthesizer: jtlv
parser: ltl

CurrentConfigName:
test

Customs: # List of custom propositions

RegionFile: # Relative path of region description file
Tutorial.regions

Sensors: # List of sensor propositions and their state (enabled = 1, disabled = 0)


======== SPECIFICATION ========

RegionMapping: # Mapping between region names and their decomposed counterparts
r2 = p2
r3 = p1
r1 = p3
others = 

Spec: # Specification in structured English
--
s.r1

[]<>s.r1
[]<>s.r2
[]<>s.r3

