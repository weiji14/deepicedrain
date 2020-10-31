# language: en
Feature: Mapping Antarctic subglacial lakes
  In order to understand the flow of subglacial water in Antarctica
  As a glaciologist,
  We want to see how active subglacial lakes are behaving over time

  Scenario Outline: Subglacial Lake Finder
    Given some altimetry data at <placename>
    When it is passed through an unsupervised clustering algorithm
    Then <this_many> potential subglacial lakes are found

    Examples:
    |  placename           | this_many |
    |  whillans_downstream | 7         |
