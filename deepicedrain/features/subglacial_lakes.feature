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
    Given some altimetry data at <location> spatially subsetted to <lake_name> with <lake_id>
    When it is turned into a spatiotemporal cube over ICESat-2 cycles <cycles>
    And visualized at each cycle using a 3D perspective at <azimuth> and <elevation>
    Then the result is an animation of ice surface elevation changing over time

    Examples:
    | location            | lake_name                      | lake_id | cycles | azimuth | elevation |
    # | whillans_downstream | Mercer 2b                      | 21      | 3-8    | 157.5   | 45        |
    # | whillans_downstream | Lower Subglacial Lake Conway   | 40      | 3-8    | 157.5   | 45        |
    # | whillans_downstream | Subglacial Lake Conway         | 41      | 3-8    | 157.5   | 45        |
    # | whillans_downstream | Subglacial Lake Whillans       | 48      | 3-8    | 157.5   | 45        |
    # | whillans_upstream   | Whillans IX                    | 50      | 3-8    | 157.5   | 45        |
    # | whillans_upstream   | Kamb 8                         | 61      | 3-8    | 157.5   | 45        |
    # | whillans_upstream   | Kamb 1                         | 62      | 3-8    | 157.5   | 45        |
    | whillans_upstream   | Kamb 34                        | 63      | 4-7    | 157.5   | 45        |
    # | siple_coast         | Kamb 12                        | 65      | 3-8    | 157.5   | 45        |
    # | siple_coast         | MacAyeal 1                     | 97      | 3-8    | 157.5   | 60        |
    # | slessor_downstream  | Slessor 45                     | 109     | 3-8    | 202.5   | 60        |
    # | slessor_downstream  | Slessor 23                     | 116     | 3-8    | 202.5   | 60        |
    | slessor_downstream  | Recovery IV                    | 141     | 3-8    | 247.5   | 45        |
    # | slessor_downstream  | Recovery 2                     | 156     | 3-8    | 202.5   | 45        |
