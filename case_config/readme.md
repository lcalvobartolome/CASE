# EWB_CONFIG

This folder is mapped to the containers `ewb_restapi` and `ewb_inferencer` so both of all the configuration variables necessary for the proper functitioning of the Evaluation Workbench.

The main variables that may need to be configured is those associated with the corpus being indexed. For each corpus that will be available at the EWB, the `config.cf` file available at the current directory should contain a section named following the name convention ``{corpus_name}-config``, and it should, at least, contain the following information:

* ``id_field``: name of the field associated to the ID in the raw corpus file.
* ``title_field``: name of the field associated to the title in the raw corpus file.
* ``date_field``: name of the field associated to the date information.

An example for the CORDIS dataset would be as follows:

```python
[cordis-config]
id_field=id
title_field=title
date_field=startDate
```
