
# Notes

### Loading libraries

``` r
knitr::opts_chunk$set(echo = TRUE, cache = FALSE)
```

``` r
library(tidyverse)
```

    ## -- Attaching packages --------------------------------------- tidyverse 1.3.1 --

    ## v ggplot2 3.3.3     v purrr   0.3.4
    ## v tibble  3.1.0     v dplyr   1.0.5
    ## v tidyr   1.1.3     v stringr 1.4.0
    ## v readr   1.4.0     v forcats 0.5.1

    ## Warning: package 'tibble' was built under R version 4.0.5

    ## Warning: package 'tidyr' was built under R version 4.0.5

    ## Warning: package 'dplyr' was built under R version 4.0.5

    ## -- Conflicts ------------------------------------------ tidyverse_conflicts() --
    ## x dplyr::filter() masks stats::filter()
    ## x dplyr::lag()    masks stats::lag()

``` r
library(tidytext)
```

    ## Warning: package 'tidytext' was built under R version 4.0.5

``` r
library(tidymodels)
```

    ## Warning: package 'tidymodels' was built under R version 4.0.5

    ## -- Attaching packages -------------------------------------- tidymodels 0.1.3 --

    ## v broom        0.7.6      v rsample      0.0.9 
    ## v dials        0.0.9      v tune         0.1.5 
    ## v infer        0.5.4      v workflows    0.2.2 
    ## v modeldata    0.1.0      v workflowsets 0.0.2 
    ## v parsnip      0.1.5      v yardstick    0.0.8 
    ## v recipes      0.1.16

    ## Warning: package 'broom' was built under R version 4.0.5

    ## Warning: package 'dials' was built under R version 4.0.5

    ## Warning: package 'infer' was built under R version 4.0.5

    ## Warning: package 'modeldata' was built under R version 4.0.5

    ## Warning: package 'parsnip' was built under R version 4.0.5

    ## Warning: package 'recipes' was built under R version 4.0.5

    ## Warning: package 'rsample' was built under R version 4.0.5

    ## Warning: package 'tune' was built under R version 4.0.5

    ## Warning: package 'workflows' was built under R version 4.0.5

    ## Warning: package 'workflowsets' was built under R version 4.0.5

    ## Warning: package 'yardstick' was built under R version 4.0.5

    ## -- Conflicts ----------------------------------------- tidymodels_conflicts() --
    ## x scales::discard() masks purrr::discard()
    ## x dplyr::filter()   masks stats::filter()
    ## x recipes::fixed()  masks stringr::fixed()
    ## x dplyr::lag()      masks stats::lag()
    ## x yardstick::spec() masks readr::spec()
    ## x recipes::step()   masks stats::step()
    ## * Use tidymodels_prefer() to resolve common conflicts.

``` r
user_reviews <- readr::read_tsv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2020/2020-05-05/user_reviews.tsv')
```

    ## 
    ## -- Column specification --------------------------------------------------------
    ## cols(
    ##   grade = col_double(),
    ##   user_name = col_character(),
    ##   text = col_character(),
    ##   date = col_date(format = "")
    ## )

``` r
user_reviews %>% head()
```

    ## # A tibble: 6 x 4
    ##   grade user_name   text                                              date      
    ##   <dbl> <chr>       <chr>                                             <date>    
    ## 1     4 mds27272    My gf started playing before me. No option to cr~ 2020-03-20
    ## 2     5 lolo2178    While the game itself is great, really relaxing ~ 2020-03-20
    ## 3     0 Roachant    My wife and I were looking forward to playing th~ 2020-03-20
    ## 4     0 Houndf      We need equal values and opportunities for all p~ 2020-03-20
    ## 5     0 ProfessorF~ BEWARE!  If you have multiple people in your hou~ 2020-03-20
    ## 6     0 tb726       The limitation of one island per Switch (not per~ 2020-03-20

### Partition Data

``` r
set.seed(42)
tidy_data <- user_reviews %>% select(-user_name)
tidy_split <- initial_split(tidy_data, p = .8)
tidy_train <- training(tidy_split)
tidy_test <- testing(tidy_split)

dim(tidy_data)
```

    ## [1] 2999    3

``` r
dim(tidy_train)
```

    ## [1] 2400    3

``` r
dim(tidy_test)
```

    ## [1] 599   3

### Text pre-processing

``` r
library(textrecipes)
```

    ## Warning: package 'textrecipes' was built under R version 4.0.5

``` r
text_recipe <- recipe(grade~text, data = tidy_train) %>% 
  step_tokenize(text) %>% 
  step_stopwords(text) %>% 
  step_tokenfilter(text, max_tokens = 500) %>% 
  step_tf(text)

text_prep <- text_recipe %>% prep()
cross_validation <- vfold_cv(tidy_train, v = 10)

# define workflow
wf <- workflow() %>% 
  add_recipe(text_recipe)
```

``` r
# each row here represents a column 2400 x 1
str(bake(text_prep, tidy_train), list.len = 15)
```

    ## tibble [2,400 x 501] (S3: tbl_df/tbl/data.frame)
    ##  $ grade                 : num [1:2400] 4 0 0 0 0 0 0 1 0 0 ...
    ##  $ tf_text_0             : num [1:2400] 0 0 0 1 0 0 0 0 0 0 ...
    ##  $ tf_text_1             : num [1:2400] 0 0 0 3 0 0 0 2 0 0 ...
    ##  $ tf_text_10            : num [1:2400] 0 0 0 1 0 0 0 0 0 0 ...
    ##  $ tf_text_2             : num [1:2400] 0 0 0 5 0 0 0 2 0 0 ...
    ##  $ tf_text_2020          : num [1:2400] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ tf_text_2nd           : num [1:2400] 2 0 0 0 0 0 0 0 0 0 ...
    ##  $ tf_text_3             : num [1:2400] 0 0 2 0 0 0 0 0 0 0 ...
    ##  $ tf_text_4             : num [1:2400] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ tf_text_5             : num [1:2400] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ tf_text_60            : num [1:2400] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ tf_text_8             : num [1:2400] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ tf_text_9             : num [1:2400] 0 0 0 0 0 0 0 0 0 0 ...
    ##  $ tf_text_ability       : num [1:2400] 0 0 0 0 0 1 0 0 0 0 ...
    ##  $ tf_text_able          : num [1:2400] 0 0 2 0 0 0 0 1 0 0 ...
    ##   [list output truncated]

### Defining Lasso Model

run parallelisation for tuning!

``` r
doParallel::registerDoParallel()

lasso_model <- linear_reg(penalty = tune(), mixture = 1) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet")

lasso_grid <- grid_regular(penalty(), levels = 10)

lasso_tune <- tune_grid(
  wf %>% add_model(lasso_model),
  resamples = cross_validation,
  grid = lasso_grid
)
```

### Understanding variable importance

``` r
lasso_model_1 <- linear_reg(penalty = 0.1 , mixture = 1) %>% 
  set_mode("regression") %>% 
  set_engine("glmnet")

lasso_fit <- wf %>% 
              add_model(lasso_model_1) %>% 
              fit(data = tidy_train)
lasso_fit %>% class()
```

    ## [1] "workflow"

``` r
lasso_fit %>% 
  pull_workflow_fit() %>% 
  tidy() %>% 
  filter(estimate > 0.15) %>% 
  mutate(term = str_remove(term, "tf_text_")) %>% 
  arrange(-estimate)
```

    ## Warning: package 'glmnet' was built under R version 4.0.5

    ## Loading required package: Matrix

    ## 
    ## Attaching package: 'Matrix'

    ## The following objects are masked from 'package:tidyr':
    ## 
    ##     expand, pack, unpack

    ## Loaded glmnet 4.1-1

    ## # A tibble: 32 x 3
    ##    term        estimate penalty
    ##    <chr>          <dbl>   <dbl>
    ##  1 (Intercept)    4.63      0.1
    ##  2 fantastic      1.13      0.1
    ##  3 amazing        0.805     0.1
    ##  4 relaxing       0.768     0.1
    ##  5 complaining    0.734     0.1
    ##  6 charming       0.731     0.1
    ##  7 awesome        0.722     0.1
    ##  8 recommend      0.712     0.1
    ##  9 perfect        0.689     0.1
    ## 10 great          0.615     0.1
    ## # ... with 22 more rows

``` r
lasso_tune %>% collect_metrics()
```

    ## # A tibble: 20 x 7
    ##          penalty .metric .estimator   mean     n std_err .config              
    ##            <dbl> <chr>   <chr>       <dbl> <int>   <dbl> <chr>                
    ##  1 0.0000000001  rmse    standard   4.25      10 0.354   Preprocessor1_Model01
    ##  2 0.0000000001  rsq     standard   0.284     10 0.0217  Preprocessor1_Model01
    ##  3 0.00000000129 rmse    standard   4.25      10 0.354   Preprocessor1_Model02
    ##  4 0.00000000129 rsq     standard   0.284     10 0.0217  Preprocessor1_Model02
    ##  5 0.0000000167  rmse    standard   4.25      10 0.354   Preprocessor1_Model03
    ##  6 0.0000000167  rsq     standard   0.284     10 0.0217  Preprocessor1_Model03
    ##  7 0.000000215   rmse    standard   4.25      10 0.354   Preprocessor1_Model04
    ##  8 0.000000215   rsq     standard   0.284     10 0.0217  Preprocessor1_Model04
    ##  9 0.00000278    rmse    standard   4.25      10 0.354   Preprocessor1_Model05
    ## 10 0.00000278    rsq     standard   0.284     10 0.0217  Preprocessor1_Model05
    ## 11 0.0000359     rmse    standard   4.25      10 0.354   Preprocessor1_Model06
    ## 12 0.0000359     rsq     standard   0.284     10 0.0217  Preprocessor1_Model06
    ## 13 0.000464      rmse    standard   4.25      10 0.354   Preprocessor1_Model07
    ## 14 0.000464      rsq     standard   0.284     10 0.0217  Preprocessor1_Model07
    ## 15 0.00599       rmse    standard   4.10      10 0.347   Preprocessor1_Model08
    ## 16 0.00599       rsq     standard   0.303     10 0.0233  Preprocessor1_Model08
    ## 17 0.0774        rmse    standard   3.73      10 0.286   Preprocessor1_Model09
    ## 18 0.0774        rsq     standard   0.353     10 0.0283  Preprocessor1_Model09
    ## 19 1             rmse    standard   4.33      10 0.0414  Preprocessor1_Model10
    ## 20 1             rsq     standard   0.0699    10 0.00784 Preprocessor1_Model10

``` r
lasso_tune %>% 
  collect_metrics() %>% 
  ggplot(aes(x = penalty, y = mean)) + geom_line() + facet_wrap(~.metric, scales = "free")
```

![](Notes_files/figure-gfm/unnamed-chunk-10-1.png)<!-- -->

### Final Workflow

``` r
lasso_best_tune <- lasso_tune %>% select_best("rmse")
final_lasso_model <- finalize_model(lasso_model, lasso_best_tune) 

lasso_wf <- finalize_workflow(
  wf %>% add_model(final_lasso_model),
  lasso_best_tune
)

lasso_wf %>% 
  fit(tidy_train) %>%
  pull_workflow_fit() %>% 
  tidy() %>% 
  filter(estimate > 0.15) %>% 
  mutate(term = str_remove(term, "tf_text_")) %>% 
  arrange(-estimate)
```

    ## # A tibble: 40 x 3
    ##    term        estimate penalty
    ##    <chr>          <dbl>   <dbl>
    ##  1 (Intercept)    4.65   0.0774
    ##  2 fantastic      1.17   0.0774
    ##  3 amazing        0.821  0.0774
    ##  4 awesome        0.814  0.0774
    ##  5 recommend      0.798  0.0774
    ##  6 relaxing       0.757  0.0774
    ##  7 complaining    0.754  0.0774
    ##  8 charming       0.725  0.0774
    ##  9 perfect        0.689  0.0774
    ## 10 juego          0.664  0.0774
    ## # ... with 30 more rows

### Final Results

``` r
lasso_eval <- lasso_wf %>% last_fit(tidy_split)
lasso_eval %>% collect_metrics()
```

    ## # A tibble: 2 x 4
    ##   .metric .estimator .estimate .config             
    ##   <chr>   <chr>          <dbl> <chr>               
    ## 1 rmse    standard       3.47  Preprocessor1_Model1
    ## 2 rsq     standard       0.359 Preprocessor1_Model1

### Final analysis on variable importance
