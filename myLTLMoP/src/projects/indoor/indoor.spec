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
parser: ltl

CurrentConfigName:
ROS

Customs: # List of custom propositions

RegionFile: # Relative path of region description file
indoor.regions

Sensors: # List of sensor propositions and their state (enabled = 1, disabled = 0)


======== SPECIFICATION ========

RegionMapping: # Mapping between region names and their decomposed counterparts
r6 = p1
r5 = p3
r4 = p4, p5
r2 = p18, p19, p20
r3 = p16, p17
r1 = p21, p22
others = 

Spec: # Specification in structured English
--
# Initialize
s.r1

# Transition
[] (s.r1 -> (next(s.r1) | next(s.r2) | next(s.r3)))
[] (s.r2 -> (next(s.r2) | next(s.r1)))
[] (s.r3 -> (next(s.r1) | next(s.r3) | next(s.r5)))
[] (s.r4 -> (next(s.r4) | next(s.r6)))
[] (s.r5 -> (next(s.r3) | next(s.r5) | next(s.r6)))
[] (s.r6 -> (next(s.r4) | next(s.r5) | next(s.r6)))

# Goal
[]<> (s.r4)

