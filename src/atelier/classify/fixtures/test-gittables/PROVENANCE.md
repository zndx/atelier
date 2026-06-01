<!-- Copyright (c) 2026 Cloudera, Inc.  All rights reserved. -->

# test-gittables Fixture Provenance

PUBLIC, non-target-domain fixture for the SVM/maxsim critical-path
test, organized for **CCO coverage** (see
docs/src/architecture/cco-coverage.md). Source: the **GitTables**
corpus (per-column DBpedia semantic types from each table's embedded
`gittables` metadata), scanned strided across the full corpus. Types
are **DBpedia ontology** classes/properties; each leaf is tagged with
its referent **CCO module** and **ICE trichotomy** class. Per-table
data carries its own upstream license (below). No customer/UAT data —
by construction (public GitTables only; answer-key names excluded).

- CCO modules covered: 10/11 (Agent, Artifact, Currency Unit, Event, Extended Relation, Facility, Geospatial, Information Entity, Quality, Time)
- leaf types: 30
- train rows: 210  |  held-out: 120 (covered: 57, weak: 63)

## Leaf types → CCO module / ICE class / DBpedia IRI

| Code | Label | CCO module | ICE class | DBpedia IRI |
|---|---|---|---|---|
| `GT.AGENT.ARTIST` | artist | Agent | DesignativeICE | http://dbpedia.org/ontology/artist |
| `GT.AGENT.AUTHOR` | author | Agent | DesignativeICE | http://dbpedia.org/ontology/author |
| `GT.AGENT.ORGANISATION` | organisation | Agent | DesignativeICE | http://dbpedia.org/ontology/organisation |
| `GT.AGENT.PERSON` | person | Agent | DesignativeICE | http://dbpedia.org/ontology/person |
| `GT.ARTIFACT.BRAND` | brand | Artifact | DesignativeICE | http://dbpedia.org/ontology/brand |
| `GT.ARTIFACT.MODEL` | model | Artifact | DesignativeICE | http://dbpedia.org/ontology/model |
| `GT.ARTIFACT.PRODUCT` | product | Artifact | DesignativeICE | http://dbpedia.org/ontology/product |
| `GT.CUR.COST` | cost | Currency Unit | DescriptiveICE | http://dbpedia.org/ontology/cost |
| `GT.CUR.CURRENCY` | currency | Currency Unit | DesignativeICE | http://dbpedia.org/ontology/currency |
| `GT.CUR.PRICE` | price | Currency Unit | DescriptiveICE | http://dbpedia.org/ontology/price |
| `GT.EVENT.EVENT` | event | Event | DesignativeICE | http://dbpedia.org/ontology/event |
| `GT.FACILITY.BUILDING` | building | Facility | DesignativeICE | http://dbpedia.org/ontology/building |
| `GT.GEO.ADDRESS` | address | Geospatial | DesignativeICE | http://dbpedia.org/ontology/address |
| `GT.GEO.CITY` | city | Geospatial | DesignativeICE | http://dbpedia.org/ontology/city |
| `GT.GEO.PLACE` | place | Geospatial | DesignativeICE | http://dbpedia.org/ontology/place |
| `GT.GEO.STATE` | state | Geospatial | DesignativeICE | http://dbpedia.org/ontology/state |
| `GT.INFO.DESCRIPTION` | description | Information Entity | DescriptiveICE | http://dbpedia.org/ontology/description |
| `GT.INFO.ID` | id | Information Entity | DesignativeICE | http://dbpedia.org/ontology/id |
| `GT.INFO.NAME` | name | Information Entity | DesignativeICE | http://dbpedia.org/ontology/name |
| `GT.INFO.TITLE` | title | Information Entity | DesignativeICE | http://dbpedia.org/ontology/title |
| `GT.QUAL.LENGTH` | length | Quality | DescriptiveICE | http://dbpedia.org/ontology/length |
| `GT.QUAL.TEMPERATURE` | temperature | Quality | DescriptiveICE | http://dbpedia.org/ontology/temperature |
| `GT.QUAL.WEIGHT` | weight | Quality | DescriptiveICE | http://dbpedia.org/ontology/weight |
| `GT.QUAL.WIDTH` | width | Quality | DescriptiveICE | http://dbpedia.org/ontology/width |
| `GT.REL.CURRENCY` | currency | Extended Relation | None | https://dbpedia.org/ontology/currency |
| `GT.REL.PUBLICATIONDATE` | publicationDate | Extended Relation | None | https://dbpedia.org/ontology/publicationDate |
| `GT.REL.PUBLISHER` | publisher | Extended Relation | None | https://dbpedia.org/ontology/publisher |
| `GT.TIME.DATE` | date | Time | DescriptiveICE | http://dbpedia.org/ontology/date |
| `GT.TIME.TIME` | time | Time | DescriptiveICE | http://dbpedia.org/ontology/time |
| `GT.TIME.YEAR` | year | Time | DescriptiveICE | http://dbpedia.org/ontology/year |

## Per-table upstream source + license

| Table ID | License | Source CSV |
|---|---|---|
| 1090 | MIT License | https://github.com/ruddfawcett/spotify-data/blob/e0c4dd127fbcdefd0fa4b706ec6bd5a39f433942/data/billboard-charts/1958-12-20.csv |
| 1099 | MIT License | https://github.com/ruddfawcett/billboard-hot-100/blob/9d7b6f119220bec3a6028b7f3f8b9f09357ad63f/charts/1986-04-26.csv |
| 11034 | MIT License | https://github.com/ruddfawcett/billboard-hot-100/blob/9d7b6f119220bec3a6028b7f3f8b9f09357ad63f/charts/1994-02-12.csv |
| 11313 | MIT License | https://github.com/cccadet/DataEngineering/blob/ebba6bc75db234265769d822d9183b3afb737304/Project%2002%20-%20Data%20Modeling%20with%20Cassandra/event_data/2018-11-11-events.csv |
| 1177 | MIT License | https://github.com/ruddfawcett/spotify-data/blob/e0c4dd127fbcdefd0fa4b706ec6bd5a39f433942/data/billboard-charts/1959-02-28.csv |
| 11851 | MIT License | https://github.com/WillAbides/klbjplaylist/blob/d5e149116a16d90d1315e06be7a165c31e0a340e/data/plays-2021-04-21.csv |
| 11993 | MIT License | https://github.com/ruddfawcett/spotify-data/blob/e0c4dd127fbcdefd0fa4b706ec6bd5a39f433942/data/billboard-charts/1988-08-06.csv |
| 10009 | Creative Commons Zero v1.0 Universal | https://github.com/rogerhyam/open_plant_names/blob/a90878a8fa8421a17b8feccb0e13250e51c0d49f/data/names/B/Boerlagea.csv |
| 10022 | MIT License | https://github.com/alixaxel/dump.HN/blob/11fd7291c89c68daf4196a47e8cfa47e4df30c8d/data/items/2009/02/20/07-08.csv |
| 100295 | Creative Commons Zero v1.0 Universal | https://github.com/rogerhyam/open_plant_names/blob/a90878a8fa8421a17b8feccb0e13250e51c0d49f/data/names/G/Gymnosteris.csv |
| 10117 | Creative Commons Zero v1.0 Universal | https://github.com/rogerhyam/open_plant_names/blob/a90878a8fa8421a17b8feccb0e13250e51c0d49f/data/names/K/Kulkarniella.csv |
| 10028 | MIT License | https://github.com/sopeeweje/NLP-AI-Medicine/blob/be9d2ccf9db1461569b7ca7f05242799dfefe38a/results/07-11-2021--155122/clusters_test/cluster-17.csv |
| 102530 | MIT License | https://github.com/digital-land/brownfield-land-pipeline/blob/04613fc57efe205239056739629a969b387526ef/var/harmonised/964617aa91b0152ff3e8a90e896df6b4ebc95c4add8a5f7bbcccc4151109d360.csv |
| 12100 | MIT License | https://github.com/hiteeshpm/Business_Intelligence_and_Business_Analysis/blob/0d82a8b58a724c2cbfaa7533f25b4db9806b6cdd/Final%20Datasets/Data%2017-18/Order%2017-18.csv |
| 138459 | MIT License | https://github.com/digital-land/brownfield-land-collection-old/blob/e9c1a7136485e2d84140a15e7315f10159908c68/var/harmonised/9a31213fcec9f37a7431258ed42e8998d9e90f6929bba9e3372a7d21c923169a.csv |
| 144351 | MIT License | https://github.com/digital-land/brownfield-land-pipeline/blob/04613fc57efe205239056739629a969b387526ef/var/mapped/6c30c1b1e4a2b292520dd8516a63c2fc0bd0925073b19d424289234b59c39c4f.csv |
| 1083 | GNU General Public License v3.0 | https://github.com/smart-facility/SPGen/blob/bf923e759365f131686cf3271cf54b974961cfa5/input%20tables/inputTables/hhRelFemale/1272706_hhRelFemale.csv |
| 1103 | GNU General Public License v3.0 | https://github.com/smart-facility/SPGen/blob/bf923e759365f131686cf3271cf54b974961cfa5/input%20tables/inputTables/hhRelFemale/1260323_hhRelFemale.csv |
| 11108 | Creative Commons Attribution Share Alike 4.0 International | https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction/blob/3b979524b2ea3924fbfbf162b2eef8bfddb7d831/MAEC_Dataset_Person_Label/20160831_OXM/text.csv |
| 1114 | GNU General Public License v3.0 | https://github.com/smart-facility/SPGen/blob/bf923e759365f131686cf3271cf54b974961cfa5/input%20tables/inputTables/hhRelFemale/1162112_hhRelFemale.csv |
| 1156 | GNU General Public License v3.0 | https://github.com/smart-facility/SPGen/blob/bf923e759365f131686cf3271cf54b974961cfa5/input%20tables/inputTables/hhRelMale/1411806_hhRelMale.csv |
| 1226 | GNU General Public License v3.0 | https://github.com/smart-facility/SPGen/blob/bf923e759365f131686cf3271cf54b974961cfa5/input%20tables/inputTables/hhRelMale/1431504_hhRelMale.csv |
| 12915 | Creative Commons Attribution Share Alike 4.0 International | https://github.com/Earnings-Call-Dataset/MAEC-A-Multimodal-Aligned-Earnings-Conference-Call-Dataset-for-Financial-Risk-Prediction/blob/3b979524b2ea3924fbfbf162b2eef8bfddb7d831/MAEC_Dataset_Person_Label/20160728_IDA/text.csv |
| 128861 | MIT License | https://github.com/swarm64/tpc-toolkit/blob/40182495bf1695164d514d24a4d93fedebd5e89b/correctness_results/tpcds/sf100/52.csv |
| 135903 | The Unlicense | https://github.com/wushu06/docker-lamp-wordpress/blob/97491a5e895608480efd450e7c76237c4405f7a0/www/wp-content/plugins/hook-me-up/Upload/1519141640_allp.csv |
| 139262 | GNU General Public License v3.0 | https://github.com/ajayneo/ebazaar/blob/7d98be7a2bcaae1d6f2359668c54ff0d7034fab2/var/import/import-20180604141101-1_48Qc_Laptop.csv |
| 157895 | MIT License | https://github.com/swarm64/tpc-toolkit/blob/40182495bf1695164d514d24a4d93fedebd5e89b/correctness_results/tpcds/sf100/19.csv |
| 182858 | GNU General Public License v3.0 | https://github.com/ajayneo/ebazaar/blob/7d98be7a2bcaae1d6f2359668c54ff0d7034fab2/var/import/import-20180809171104-1_QCNBK-B_Laptop_code8.8.csv |
| 0 | GNU General Public License v3.0 | https://github.com/guohuadeng/odoo12-x64/blob/a0aad3d6e23771630d05bd5c6c53cf8d758ca9f9/source/odoo/addons/google_calendar/security/ir.model.access.csv |
| 10041 | GNU Affero General Public License v3.0 | https://github.com/clickode/l10n-italy/blob/2205dcc9421f4f4f23dadc567d161fbd08e02295/l10n_it_reverse_charge/security/ir.model.access.csv |
| 101181 | GNU Affero General Public License v3.0 | https://github.com/ubic135/odoo-design/blob/b2774f961b9d7f758dff766dab17d16931653f90/addons/crm/security/ir.model.access.csv |
| 102987 | GNU Affero General Public License v3.0 | https://github.com/darkleons/odoo/blob/8d077838b7c55892e82d69a2a54d9c94838bbd6f/addons/crm/security/ir.model.access.csv |
| 1038 | GNU Affero General Public License v3.0 | https://github.com/WytheLi/openerp-7.0-20170329/blob/84ec3639897e13f536d4ebba8bed0e06f0aaf6e4/openerp/addons/portal_project_long_term/security/ir.model.access.csv |
| 10464 | GNU Affero General Public License v3.0 | https://github.com/dejankosutic/conformio/blob/4744a50a21e817ed8373548178c941567203344e/document_webdav_fast/security/ir.model.access.csv |
| 10787 | MIT License | https://github.com/ShaheenHossain/itpp-labs_pos-addons/blob/8c5047af10447eb3d137c84111127fae1a8970b6/pos_debt_notebook/security/ir.model.access.csv |
| 111885 | MIT License | https://github.com/SBRG/modulytics/blob/8ce05a7986211784264e097fd6f890481d72dc34/organisms/e_coli/precise1/iModulon_files/54/54_gene_table.csv |
| 11619 | MIT License | https://github.com/stephanie-shields/foodsales-app-design/blob/08e2145e88d36a7cee941b0d5bd49b4dc93b6bbe/data/FoodSales.csv |
| 1192 | MIT License | https://github.com/jeromebailey/sir/blob/608c9012a418dc1ba7ce5a84b92e64a89914c5ff/uploads/product-stock-level15.csv |
| 13718 | MIT License | https://github.com/ZZR0/ISSTA21-JIT-DP/blob/8e7874ec9b983fcb45b8a4ebe5222721ffe75da5/Data_Extraction/git_base/issue_datasets/platform/Mylyn%20Commons_2004-01-01_2018-01-01.csv |
| 10177 | MIT License | https://github.com/randomguy4214/poor-almanac-5/blob/fad995c0a9057d7fea0f8dea67c6db98ba14427f/0_input/temp/financials/DSECF.csv |
| 10276 | MIT License | https://github.com/randomguy4214/poor-almanac-5/blob/fad995c0a9057d7fea0f8dea67c6db98ba14427f/0_input/temp/financials/FNMAO.csv |
| 10320 | MIT License | https://github.com/randomguy4214/poor-almanac-5/blob/fad995c0a9057d7fea0f8dea67c6db98ba14427f/0_input/temp/financials/ELEZY.csv |
| 10406 | MIT License | https://github.com/randomguy4214/poor-almanac-5/blob/fad995c0a9057d7fea0f8dea67c6db98ba14427f/0_input/temp/financials/UN01.DE.csv |
| 10433 | MIT License | https://github.com/randomguy4214/poor-almanac-5/blob/fad995c0a9057d7fea0f8dea67c6db98ba14427f/0_input/temp/financials/HIMX.csv |
| 10554 | MIT License | https://github.com/randomguy4214/poor-almanac-5/blob/fad995c0a9057d7fea0f8dea67c6db98ba14427f/0_input/temp/financials/EGHT.csv |
| 10577 | MIT License | https://github.com/randomguy4214/poor-almanac-5/blob/fad995c0a9057d7fea0f8dea67c6db98ba14427f/0_input/temp/financials/HZN.csv |
| 10052 | Academic Free License v3.0 | https://github.com/cshjin/Microgrid_sim/blob/7c032dc3d11c563afaebda744c9405b13b85e76e/Data/weather_data/KMDW/wunderground_1993_10_05.csv |
| 100978 | MIT License | https://github.com/itzdan/Azure-Zentinel/blob/be76ef747e782c8beb17d3ac5b189b61db9bf4e3/Sample%20Data/Custom/ZimperiumMitigationLog_CL.csv |
| 10290 | Academic Free License v3.0 | https://github.com/cshjin/Microgrid_sim/blob/7c032dc3d11c563afaebda744c9405b13b85e76e/Data/weather_data/KMDW/wunderground_1994_07_07.csv |
| 10460 | Academic Free License v3.0 | https://github.com/cshjin/Microgrid_sim/blob/7c032dc3d11c563afaebda744c9405b13b85e76e/Data/weather_data/KMDW/wunderground_1996_07_28.csv |
| 10627 | Academic Free License v3.0 | https://github.com/cshjin/Microgrid_sim/blob/7c032dc3d11c563afaebda744c9405b13b85e76e/Data/weather_data/KMDW/wunderground_1992_02_26.csv |
| 10728 | Academic Free License v3.0 | https://github.com/cshjin/Microgrid_sim/blob/7c032dc3d11c563afaebda744c9405b13b85e76e/Data/weather_data/KMDW/wunderground_1999_07_10.csv |
| 10934 | Academic Free License v3.0 | https://github.com/cshjin/Microgrid_sim/blob/7c032dc3d11c563afaebda744c9405b13b85e76e/Data/weather_data/KMDW/wunderground_1991_09_11.csv |
| 139578 | BSD 3-Clause "New" or "Revised" License | https://github.com/OasisLMF/OasisLMF_SQL/blob/4c0edef7b346cf2a0b3cd0813320d063fa3e8b40/tests/model_preparation/examples/single_acc_level_SS_all_risks/location.csv |
| 19919 | MIT License | https://github.com/jlcatonjr/Learn-Python-for-Stats-and-Econ/blob/6b47325fb158615bcc10d0a7ec7d28257d638089/In%20Class%20Projects/In%20Class%20Examples%20Spring%202019/Section%208/County/Planning%20Region%202.csv |
| 10315 | MIT License | https://github.com/mishrabp/node-northwind-app/blob/0532c9c11a40df092087122657137dc787a746c5/dal/csvdata/employees.csv |
| 1035 | MIT License | https://github.com/S2-group/icse-seip-2020-replication-package/blob/2f4453b4278c18c1c37641267a7eb9ab6fb17e03/online_questionnaire/online_questionnaire_scripts/Mail%20Sender%20/emails.csv |
| 10447 | Apache License 2.0 | https://github.com/afermon/Freelancr/blob/0d27632582cdcdb1ac6840aa8ad47a26375839e5/src/main/resources/config/liquibase/user_freelancr.csv |
| 11782 | Apache License 2.0 | https://github.com/kylehounslow/mlbootcampSF/blob/9628dd99ba34009c166d13f26261d936641970e6/notebooks/data/sf/mar10_2018/properties-94118.csv |
| 121418 | GNU Affero General Public License v3.0 | https://github.com/omusico/SelkirkCRM/blob/f2d551a7b37cec4e244daa3ae609524736d51387/SelkirkData/Export/Selkirk_Client_imp_tpl.csv |
| 122432 | GNU General Public License v3.0 | https://github.com/ari-dasci/S-SDG-Decision/blob/c3211c90697b69db25f6f76d870a7953dccde8ff/exports/ODS-12/wikisurvey_19765_nonvotes_2021-01-14T16_48_44Z.csv |
| 12600 | The Unlicense | https://github.com/open-austin/construction-permits/blob/0464af73e913012b56518a8a1e5b3b641c367a99/data/1999/1999-06-29.csv |
| 10382 | MIT License | https://github.com/RCN/events-scraper/blob/e2146c977e00931e20e2545706d3aabe1c071d40/data/results/permit-2014-1639_plain.csv |
| 1291 | MIT License | https://github.com/RCN/events-scraper/blob/e2146c977e00931e20e2545706d3aabe1c071d40/data/results/permit-2013-1873_plain.csv |
| 13079 | MIT License | https://github.com/terminological/bibliographic-api-client/blob/02bc088e6b514b8001d5273f45982d3b68966fc8/src/test/resources/modelReferences.csv |
| 13816 | MIT License | https://github.com/andreasagap/Flask_WebApp_TripAdvisor/blob/1899921296ef46628c532ffc130c2c0f065bb9d5/user_profiles/joyhuang2016.csv |
| 1 | BSD 3-Clause "New" or "Revised" License | https://github.com/NREL/hasty/blob/d293056f169fedb7988d5ccc524c05a6881ef6a9/hasty/lib/ComponentPoints.csv |
| 10019 | Creative Commons Attribution Share Alike 4.0 International | https://github.com/e-tony/best-of-ml-julia/blob/11feac22316dd39dc15e8f726ef3aca060c60f61/history/2021-02-09_projects.csv |
| 1008 | MIT License | https://github.com/mora-lab/benchmarks/blob/cde839aa35d39f74d03283c8bf50be3859349174/single-sample/data/GSE35571_pdata.csv |
| 101221 | GNU Affero General Public License v3.0 | https://github.com/fvviz/Command-maker-bot/blob/36629b13e9567e444d639fcf9e10aeec4cfe855f/data/commands/embed/581084433646616576.csv |
| 1019 | GNU General Public License v3.0 | https://github.com/SunBuild/moodle-azure/blob/f1c70cf91cd07791587728dd3584f7e07b342312/group/tests/fixtures/groups_import.csv |
| 1023 | MIT License | https://github.com/tushartushar/miningSmellData/blob/fb143af48109a76075011245a7093c90081c58c2/Results/jancowol_Shovel/Designite_Shovel.Tests_ImpSmells.csv |
| 100159 | Apache License 2.0 | https://github.com/sotorrent/metric-evaluation/blob/f559874e8f331d1cd8e1d01672fd5fcfc418096a/testdata/samples_comparison/PostId_VersionCount_SO_17-06_sample_100_multiple_possible_links/files/18724229.csv |
| 10134 | MIT License | https://github.com/alixaxel/dump.HN/blob/11fd7291c89c68daf4196a47e8cfa47e4df30c8d/data/items/2008/07/28/11-12.csv |
| 10136 | MIT License | https://github.com/alixaxel/dump.HN/blob/11fd7291c89c68daf4196a47e8cfa47e4df30c8d/data/items/2008/06/20/14-15.csv |
| 10237 | MIT License | https://github.com/alixaxel/dump.HN/blob/11fd7291c89c68daf4196a47e8cfa47e4df30c8d/data/items/2008/12/02/04-05.csv |
| 1027 | GNU General Public License v2.0 | https://github.com/kevncobb/limed9/blob/eac7442fdfd1723370a29ea6b450bb9df335e380/docroot/core/profiles/demo_umami/modules/demo_umami_content/default_content/languages/en/node/recipe.csv |
| 11876 | BSD 3-Clause "New" or "Revised" License | https://github.com/salesforce/esprit/blob/83489a15be76629a7d2428cb4a47bd797b9aa621/data_tables/initial_state/val/csv/00020-007.csv |
| 11881 | BSD 3-Clause "New" or "Revised" License | https://github.com/salesforce/esprit/blob/83489a15be76629a7d2428cb4a47bd797b9aa621/data_tables/initial_state/train/csv/00020-090.csv |
| 11889 | BSD 3-Clause "New" or "Revised" License | https://github.com/salesforce/esprit/blob/83489a15be76629a7d2428cb4a47bd797b9aa621/data_tables/initial_state/train/csv/00020-048.csv |
| 10006 | Apache License 2.0 | https://github.com/sheldonsebastian/Capital-Bikeshare/blob/bf23e2438eff84ca3952edbfb3c90c6d01d36eb1/Scraping%20Logic%20and%20Preprocessing/Weather%20Data%20Scraping/weatherDataScraped2013-12-16.csv |
| 10227 | Apache License 2.0 | https://github.com/sheldonsebastian/Capital-Bikeshare/blob/bf23e2438eff84ca3952edbfb3c90c6d01d36eb1/Scraping%20Logic%20and%20Preprocessing/Weather%20Data%20Scraping/weatherDataScraped2019-09-22.csv |
| 1071 | MIT License | https://github.com/YuanSingapore/MIMIC-LSTM-Mortality-deployment/blob/56296b805a99d8aae6e67439e313278770d289e1/data/test/14592_episode1_timeseries.csv |
| 1085 | MIT License | https://github.com/YuanSingapore/MIMIC-LSTM-Mortality-deployment/blob/56296b805a99d8aae6e67439e313278770d289e1/data/test/3987_episode2_timeseries.csv |
| 1096 | MIT License | https://github.com/YuanSingapore/MIMIC-LSTM-Mortality-deployment/blob/56296b805a99d8aae6e67439e313278770d289e1/data/test/6621_episode1_timeseries.csv |
| 1126 | BSD 2-Clause "Simplified" License | https://github.com/ponsonio-aurea/silvershop-core/blob/a49b724b075633cd27a26df926d5c9b7c058c9de/tests/test_products.csv |
| 1160 | MIT License | https://github.com/YuanSingapore/MIMIC-LSTM-Mortality-deployment/blob/56296b805a99d8aae6e67439e313278770d289e1/data/test/17022_episode1_timeseries.csv |
| 1186 | GNU Affero General Public License v3.0 | https://github.com/benibienz/drawdown/blob/2d46e17ae7cfd0a86a145f489ee164575394956e/solution/recycledpaper/vma_data/Current_Adoption.csv |
| 13297 | MIT License | https://github.com/saralafia/ERI-maps/blob/78f6facfd23ecfacfb0108f86bbec61f0ac7d207/outputs/LDA/LDA-43-topic-keyword-weights.csv |
| 101489 | MIT License | https://github.com/diwu1990/uSystolic-Sim/blob/3400dd2ff9e8c25083bb5cd157bde335f52f3afc/config_src/network_config/gemm_config_from_scalesim/deepbench/DeepBenchConv/DeepBench.csv |
| 10654 | GNU General Public License v3.0 | https://github.com/brady-haffey/collector-examples/blob/800da0b591847178beede49ced372bf5ff4e4ed4/web/Default/DefaultSurveys/autism_quotient.csv |
| 11663 | Creative Commons Zero v1.0 Universal | https://github.com/preprocessed-connectomes-project/giavasis2015-QAP-paper/blob/789429fc79771ab830c0597728a6e1006be6350a/data_analysis/func_corr_scan_params.csv |
| 1182 | MIT License | https://github.com/m-hasan-n/hlp/blob/fc73f1966fe3acb178863846544abbede79d6104/dataset/sub_22/T107/scene/scene_layout_T107.csv |
| Book_1jour-1jeu.com_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_5carti.ro_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_7books.hu_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_80mundos.com_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_abbeville.com_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_abebooks.co.uk_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_abebooks.fr_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_11x17.pt_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_2014brazil.co.uk_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_24symbols.com_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_4thestate.co.uk_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_5sentidoseditora.pt_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| Book_abramsandchronicle.co.uk_September2020_CPA | WDC SOTAB v2 (research use) | https://webdatacommons.org/structureddata/sotab/ |
| 100 | GNU General Public License v3.0 | https://github.com/rvalla/SETM/blob/adc8d3e1a624be4d4ab7ce986ba3a33e7a6cd18d/HumanToHumanModel/SimulationData/Infections/02092020_5K_360d_ATST04501_0_infections.csv |
| 10075 | Apache License 2.0 | https://github.com/MayankAgarwal/DiskDriveDaysPredictor/blob/2f67a777617ae1f201648394981eeffeba50b611/Code/Data/Western%20Digital/WDC%20WD1600AAJS/WD-WMAYV2570576.csv |
| 100836 | GNU General Public License v3.0 | https://github.com/jjmcnelis/nasa-cmr-inventory/blob/36a3de753b71c559a795e61d0a4aa5756647ddd0/projects/ornldaac/atom/ATom_UHSAS_Data_1619/gr.csv |
| 100845 | MIT License | https://github.com/xasos/IlliniGuide/blob/a2695decde1479843503e52fb48677c9d75d559a/data/ReviewsCSV/CSV's/Dov_Weiss_1_reviews.csv |
| 1013 | MIT License | https://github.com/xasos/IlliniGuide/blob/a2695decde1479843503e52fb48677c9d75d559a/data/ReviewsCSV/CSV's/Alfred_Roca_1_reviews.csv |
| 10054 | MIT License | https://github.com/Eliseowzy/financialFraudDetection/blob/ddf57c2ecfab25fc2989adb0283aef56a8156a62/programm/data/email_corpus_by_person/james_foster_enron_com.csv |
| 10116 | MIT License | https://github.com/Eliseowzy/financialFraudDetection/blob/ddf57c2ecfab25fc2989adb0283aef56a8156a62/programm/data/email_corpus_by_person/eloan_mcfeely_eloan_com.csv |
| 1012 | University of Illinois/NCSA Open Source License | https://github.com/rai-project/dlperf/blob/88ce34751cf83dd3aecf5967922664341df4b2fc/assets/cudnn_advised_latency/batchsize_64/Tesla_M60/BVLC_CaffeNet.csv |
| 102187 | Creative Commons Zero v1.0 Universal | https://github.com/rogerhyam/open_plant_names/blob/a90878a8fa8421a17b8feccb0e13250e51c0d49f/data/names/T/Trichometasphaeria.csv |
| 10244 | Creative Commons Zero v1.0 Universal | https://github.com/rogerhyam/open_plant_names/blob/a90878a8fa8421a17b8feccb0e13250e51c0d49f/data/names/P/Prosphytochloa.csv |
| 10254 | Creative Commons Zero v1.0 Universal | https://github.com/rogerhyam/open_plant_names/blob/a90878a8fa8421a17b8feccb0e13250e51c0d49f/data/names/P/Psammospora.csv |
