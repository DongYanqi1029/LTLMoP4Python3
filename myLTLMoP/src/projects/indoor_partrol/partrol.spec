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
patrol.regions

Sensors: # List of sensor propositions and their state (enabled = 1, disabled = 0)


======== SPECIFICATION ========

RegionMapping: # Mapping between region names and their decomposed counterparts
dining = p22, p23, p24, p25
porch = p2
kitchen = p8
bedroom = p26, p27, p28
living = p15, p16, p17, p18, p19, p20, p21
others = 

Spec: # Specification in structured English
--

s.porch

[](s.porch -> (next(s.porch) | next(s.kitchen) | next(s.living)))
[](s.living -> (next(s.living) | next(s.porch) | next(s.dining) | next(s.bedroom)))
[](s.kitchen -> (next(s.kitchen) | next(s.porch) | next(s.dining)))
[](s.dining -> (next(s.dining) | next(s.kitchen) | next(s.living)))
[](s.bedroom -> (next(s.living) | next(s.bedroom)))

[]<> (s.porch)
[]<> (s.living)
[]<> (s.dining)
[]<> (s.kitchen)

