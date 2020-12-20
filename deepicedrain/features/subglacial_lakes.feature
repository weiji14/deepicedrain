# language: en
Feature: Mapping Antarctic subglacial lakes
  In order to understand the flow of subglacial water in Antarctica
  As a glaciologist,
  We want to see how active subglacial lakes are behaving over time

  Scenario Outline: Subglacial Lake Finder
    Given some altimetry data at <location>
    When it is passed through an unsupervised clustering algorithm
    Then <this_many> potential subglacial lakes are found

    Examples:
    |  location            | this_many |
    |  whillans_downstream | 7         |


  Scenario Outline: Subglacial Lake Animation
    Given some altimetry data over <lake_name> at <location>
    When it is turned into a spatiotemporal cube over ICESat-2 cycles <cycles>
    And visualized at each cycle using a 3D perspective at <azimuth> and <elevation>
    Then the result is an animation of ice surface elevation changing over time

    Examples:
    | lake_name    | location            | cycles | azimuth | elevation |
    # | Mercer XV    | whillans_downstream | 3-8    | 157.5   | 45        |
    # | Whillans 7   | whillans_upstream   | 3-8    | 157.5   | 45        |
    # | Whillans 6   | whillans_upstream   | 3-8    | 157.5   | 45        |
    # | Whillans X   | whillans_upstream   | 3-8    | 157.5   | 45        |
    # | Whillans XI  | whillans_upstream   | 3-8    | 157.5   | 45        |
    # | Whillans IX  | whillans_upstream   | 3-8    | 157.5   | 45        |
    # | Whillans 12  | whillans_downstream | 3-8    | 157.5   | 45        |
    # | Kamb 8       | whillans_upstream   | 3-8    | 157.5   | 45        |
    # | Kamb 1       | whillans_upstream   | 3-8    | 157.5   | 45        |
    | Kamb 34      | whillans_upstream   | 4-7    | 157.5   | 45        |
    # | Kamb 12      | siple_coast         | 3-8    | 157.5   | 45        |
    # | MacAyeal 1   | siple_coast         | 3-8    | 157.5   | 60        |
    # | Slessor 45   | slessor_downstream  | 3-8    | 202.5   | 60        |
    # | Slessor 23   | slessor_downstream  | 3-8    | 202.5   | 60        |
    | Recovery IV  | slessor_downstream  | 3-8    | 247.5   | 45        |


  Scenario Outline: Subglacial Lake Mega-Cluster Animation
    Given some altimetry data over <lake_name> at <location>
    When it is turned into a spatiotemporal cube over ICESat-2 cycles <cycles>
    And visualized at each cycle using a 3D perspective at <azimuth> and <elevation>
    Then the result is an animation of ice surface elevation changing over time

    Examples:
    | lake_name                | location            | cycles | azimuth | elevation |
    # | Lake 78                  | whillans_downstream | 3-8    | 157.5   | 45        |
    # | Subglacial Lake Conway   | whillans_downstream | 3-8    | 157.5   | 45        |
    | Subglacial Lake Mercer   | whillans_downstream | 3-8    | 157.5   | 45        |
    # | Subglacial Lake Whillans | whillans_downstream | 3-8    | 157.5   | 45        |
    # | Recovery 2               | slessor_downstream  | 3-8    | 202.5   | 45        |

  Scenario Outline: Subglacial Lake Crossover Anomalies
    Given some altimetry data over <lake_name> at <location>
    When ice surface height anomalies are calculated at crossovers within the lake area
    Then we see a trend of active subglacial lake surfaces changing over time

    Examples:
    | lake_name                | location            |
    | Whillans 7               | whillans_upstream   |
    # | Whillans IX              | whillans_upstream   |
    # | Subglacial Lake Whillans | whillans_downstream |
    | Whillans 12              | whillans_downstream |
