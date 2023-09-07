library(tidyverse)
library(tidymodels)


# Import data -------------------------------------------------------------

train_rule <- read_csv("./data/train_rule.csv")
train_device <- read_csv("./data/train_device.csv")
test_rule <- read_csv("./data/test_rule.csv")
test_device <- read_csv("./data/test_device.csv")
sample_submission <- read_csv("./data/sample_submission.csv")

gc()


# Data validation ---------------------------------------------------------

# train data
identical(train_device$user_id |> unique() |> sort(), 
          train_rule$user_id |> unique() |> sort())

identical(train_device$device_model |> unique() |> sort(), 
          train_rule$trigger_device |> unique() |> sort())

identical(train_device$device_model |> unique() |> sort(), 
          train_rule$action_device |> unique() |> sort())

# test data
identical(test_device$user_id |> unique() |> sort(), 
          test_rule$user_id |> unique() |> sort())

identical(test_device$device_model |> unique() |> sort(), 
          test_rule$trigger_device |> unique() |> sort())

identical(test_device$device_model |> unique() |> sort(), 
          test_rule$action_device |> unique() |> sort())

# train vs test data
trigger_devices <- train_rule$trigger_device |> unique() |> sort()
trigger_states <- train_rule$trigger_state |> unique() |> sort()
action_devices <- train_rule$action_device |> unique() |> sort()
actions <- train_rule$action |> unique() |> sort()

mean(test_rule$trigger_device %in% trigger_devices)
mean(test_rule$trigger_state %in% trigger_states)
mean(test_rule$action_device %in% action_devices)
mean(test_rule$action %in% actions)

mean(test_rule$user_id %in% train_rule$user_id)
mean(test_device$user_id %in% train_device$user_id)
mean(test_device$device_id %in% train_device$device_id)

# rules
train_rule |> 
  mutate_if(is.numeric, ~ str_trim(format(., scientific = FALSE))) |> 
  mutate(rule_2 = paste(trigger_device_id,
                        trigger_state_id,
                        action_id,
                        action_device_id,
                        sep = "_"),
         is_exact = rule == rule_2) |> 
  pull(is_exact) |> 
  mean()

test_rule |> 
  mutate_if(is.numeric, ~ str_trim(format(., scientific = FALSE))) |> 
  mutate(rule_2 = paste(trigger_device_id,
                        trigger_state_id,
                        action_id,
                        action_device_id,
                        sep = "_"),
         is_exact = rule == rule_2) |> 
  pull(is_exact) |> 
  mean()


# Extract mappings and unique values --------------------------------------

# mappings
trigger_device_map <- train_rule |> 
  distinct(id = trigger_device_id, device = trigger_device) |> 
  arrange(id)
trigger_state_map <- train_rule |> 
  distinct(id = trigger_state_id, state = trigger_state) |> 
  arrange(id)
action_map <- train_rule |> 
  distinct(id = action_id, action = action) |> 
  arrange(id)
action_device_map <- train_rule |> 
  distinct(id = action_device_id, device = action_device) |> 
  arrange(id)
device_map <- train_rule |> 
  distinct(device = action_device) |> 
  arrange(device) |>
  mutate(id = 1:length(device)) |> 
  select(2, 1)

# unique values
train_user_ids <- train_rule$user_id |> unique() |> sort()
test_user_ids <- test_rule$user_id |> unique() |> sort()
states <- trigger_state_map$state |> sort()
actions <- action_map$action |> sort()
devices <- device_map$device |> sort()


# Train feature extraction ------------------------------------------------

# extract 1: trigger state
tbl_trigger_state <- list()

for (i in seq_along(states)) {
  .column <- paste0("trigger_state_", i)
  tbl_trigger_state[["user_id"]] <- train_rule$user_id
  tbl_trigger_state[[.column]] <- states[i] == train_rule$trigger_state
}

tbl_trigger_state <- tbl_trigger_state |> bind_cols() 

# extract 2: trigger device
tbl_trigger_device <- list()

for (i in seq_along(devices)) {
  .column <- paste0("trigger_device_", i)
  tbl_trigger_device[["user_id"]] <- train_rule$user_id
  tbl_trigger_device[[.column]] <- devices[i] == train_rule$trigger_device
}

tbl_trigger_device <- tbl_trigger_device |> bind_cols() 

# action 3: device
tbl_action <- list()

for (i in seq_along(actions)) {
  .column <- paste0("action_", i)
  tbl_action[["user_id"]] <- train_rule$user_id
  tbl_action[[.column]] <- actions[i] == train_rule$action
}

tbl_action <- tbl_action |> bind_cols()

# extract 4: action device
tbl_action_device <- list()

for (i in seq_along(devices)) {
  .column <- paste0("action_device_", i)
  tbl_action_device[["user_id"]] <- train_rule$user_id
  tbl_action_device[[.column]] <- devices[i] == train_rule$action_device
}

tbl_action_device <- tbl_action_device |> bind_cols()

# extract 5: user devices
tbl_user_device <- train_device |> 
  count(user_id, device_model) |> 
  arrange(device_model) |> 
  pivot_wider(names_from = device_model, values_from = n) |> 
  mutate_all(~ if_else(!is.na(.), ., 0)) |> 
  rename_all(~ paste0("user_device_", pull(device_map, id, device)[.])) |>  
  rename_at(1, ~ "user_id") |>
  janitor::clean_names()

# checking
map_int(tbl_trigger_state |> select(-user_id), sum)
map_int(tbl_trigger_device |> select(-user_id), sum)
map_int(tbl_action |> select(-user_id), sum)
map_int(tbl_action_device |> select(-user_id), sum)
map_int(tbl_user_device |> select(-user_id), sum)

identical(tbl_trigger_state$user_id, tbl_trigger_device$user_id)
identical(tbl_trigger_state$user_id, tbl_action$user_id)
identical(tbl_trigger_state$user_id, tbl_action_device$user_id)

# combine features
features <- tbl_trigger_device |> 
  bind_cols(select(tbl_trigger_state, -user_id)) |> 
  bind_cols(select(tbl_action, -user_id)) |> 
  bind_cols(select(tbl_action_device, -user_id)) |> 
  inner_join(tbl_user_device, by = "user_id") |> 
  arrange(user_id) |> 
  mutate(rule_id = 1:length(user_id)) |> 
  select(rule_id, user_id, starts_with("user_device"), everything()) |> 
  mutate_at(1, as.factor) |> 
  mutate_at(2, as.factor) |> 
  mutate_if(is.logical, as.numeric)

write_csv(features, "./data/features-train.csv")


# Test feature extraction -------------------------------------------------

# extract 1: trigger state
tbl_trigger_state <- list()

for (i in seq_along(states)) {
  .column <- paste0("trigger_state_", i)
  tbl_trigger_state[["user_id"]] <- test_rule$user_id
  tbl_trigger_state[[.column]] <- states[i] == test_rule$trigger_state
}

tbl_trigger_state <- tbl_trigger_state |> bind_cols() 

# extract 2: trigger device
tbl_trigger_device <- list()

for (i in seq_along(devices)) {
  .column <- paste0("trigger_device_", i)
  tbl_trigger_device[["user_id"]] <- test_rule$user_id
  tbl_trigger_device[[.column]] <- devices[i] == test_rule$trigger_device
}

tbl_trigger_device <- tbl_trigger_device |> bind_cols() 

# action 3: device
tbl_action <- list()

for (i in seq_along(actions)) {
  .column <- paste0("action_", i)
  tbl_action[["user_id"]] <- test_rule$user_id
  tbl_action[[.column]] <- actions[i] == test_rule$action
}

tbl_action <- tbl_action |> bind_cols()

# extract 4: action device
tbl_action_device <- list()

for (i in seq_along(devices)) {
  .column <- paste0("action_device_", i)
  tbl_action_device[["user_id"]] <- test_rule$user_id
  tbl_action_device[[.column]] <- devices[i] == test_rule$action_device
}

tbl_action_device <- tbl_action_device |> bind_cols()

# extract 5: user devices
tbl_user_device <- test_device |> 
  count(user_id, device_model) |> 
  arrange(device_model) |> 
  pivot_wider(names_from = device_model, values_from = n) |> 
  mutate_all(~ if_else(!is.na(.), ., 0)) |> 
  rename_all(~ paste0("user_device_", pull(device_map, id, device)[.])) |>  
  rename_at(1, ~ "user_id") |>
  janitor::clean_names()

# checking
map_int(tbl_trigger_state |> select(-user_id), sum)
map_int(tbl_trigger_device |> select(-user_id), sum)
map_int(tbl_action |> select(-user_id), sum)
map_int(tbl_action_device |> select(-user_id), sum)
map_int(tbl_user_device |> select(-user_id), sum)

identical(tbl_trigger_state$user_id, tbl_trigger_device$user_id)
identical(tbl_trigger_state$user_id, tbl_action$user_id)
identical(tbl_trigger_state$user_id, tbl_action_device$user_id)

# combine features
features <- tbl_trigger_device |> 
  bind_cols(select(tbl_trigger_state, -user_id)) |> 
  bind_cols(select(tbl_action, -user_id)) |> 
  bind_cols(select(tbl_action_device, -user_id)) |> 
  inner_join(tbl_user_device, by = "user_id") |> 
  arrange(user_id) |> 
  mutate(rule_id = 1:length(user_id)) |> 
  select(rule_id, user_id, starts_with("user_device"), everything()) |> 
  mutate_at(1, as.factor) |> 
  mutate_at(2, as.factor) |> 
  mutate_if(is.logical, as.numeric)

write_csv(features, "./data/features-test.csv")