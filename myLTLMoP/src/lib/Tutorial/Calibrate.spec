# This is a specification definition file for the LTLMoP toolkit.
# Format details are described at the beginning of each section below.


======== SETTINGS ========

Actions: # List of action propositions and their state (enabled = 1, disabled = 0)

CompileOptions:
convexify: True
fastslow: False
symbolic: False
decompose: True
use_region_bit_encoding: True
synthesizer: jtlv
parser: structured

CurrentConfigName:
calibrate

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
go to r1
go to r2
go to r3

