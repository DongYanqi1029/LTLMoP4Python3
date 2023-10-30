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
Untitled configuration

Customs: # List of custom propositions

RegionFile: # Relative path of region description file
construct.regions

Sensors: # List of sensor propositions and their state (enabled = 1, disabled = 0)


======== SPECIFICATION ========

RegionMapping: # Mapping between region names and their decomposed counterparts
r1 = p19, p20
r5 = p1
r4 = p13, p14
r3 = p15, p16
r2 = p17, p18
others = 

Spec: # Specification in structured English
--
s.r1

[](s.r1 ->(next(s.r1) | next(s.r3) | next(s.r5)))
[](s.r2 ->(next(s.r2) | next(s.r3)))
[](s.r3 ->(next(s.r1) | next(s.r2) | next(s.r3) | next(s.r4)))
[](s.r4 ->(next(s.r3) | next(s.r4)))
[](s.r5 ->(next(s.r5) | next(s.r1)))

[]<> (s.r1)
[]<> (s.r3)
[]<> (s.r2)
[]<> (s.r4)
[]<> (s.r5)

