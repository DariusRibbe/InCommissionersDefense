
# 1 Extracting the Speeches

# Directions to folders
dir_25    <- "YOUR DIRECTION to speeches 25" # change to your direction
dir_24    <- "YOUR DIRECTION to speeches 24"
dir_trans <- "YOUR DIRECTION to translations"

csv_25    <- "corpus_eu_speeches_25.csv"
csv_24    <- "corpus_eu_speeches_24.csv"
csv_trans <- "corpus_eu_speeches_all.csv"

# Packages
if(!require(pdftools)){install.packages("pdftools")}
if(!require(stringr)){install.packages("stringr")}
if(!require(tibble)){install.packages("tibble")}
if(!require(purrr)){install.packages("purrr")}
if(!require(dplyr)){install.packages("dplyr")}
if(!require(readr)){install.packages("readr")}
if(!require(stringdist)){install.packages("stringdist")}
if(!require(stringi)){install.packages("stringi")}
if(!require(readxl)){install.packages("readxl")}
if(!require(cld3)){install.packages("cld3")}
if(!require(tidyr)){install.packages("tidyr")}
if(!require(lubridate)){install.packages("lubridate")}
if(!require(ggplot2)){install.packages("ggplot2")}
if(!require(tidyverse)){install.packages("tidyverse")}
if(!require(patchwork)){install.packages("patchwork")}
if(!require(survey)){install.packages("survey")}
if(!require(irr)){install.packages("irr")}
if(!require(writexl)){install.packages("writexl")}

# Names-Dictionary for Speaker extraction
commissioners <- read_excel("Commissioners.xlsx")
names_dict <- commissioners$Last_Name

# Speaker extraction
extract_speaker_dict <- function(heading_full, names_dict,
                                 fuzzy = TRUE, max_dist = 1) {
  
  if (is.na(heading_full) || heading_full == "") return(NA_character_)
  
  heading_norm <- stri_trans_general(heading_full, "Latin-ASCII")
  heading_norm <- str_squish(heading_norm)
  
  names_norm <- stri_trans_general(names_dict, "Latin-ASCII")
  
  names_escaped <- str_replace_all(
    names_norm,
    "([\\.^$|()*+?{}\\[\\]\\\\])", "\\\\\\1"
  )
  
  pattern <- paste0("\\b(", paste(names_escaped, collapse = "|"), ")\\b")
  
  exact_hits <- str_extract_all(
    heading_norm,
    regex(pattern, ignore_case = TRUE)
  )[[1]]
  
  if (length(exact_hits) > 0) {
    return(names_dict[match(exact_hits[1], names_norm)])
  }
  
  if (fuzzy) {
    dists <- stringdist(
      tolower(heading_norm),
      tolower(names_norm),
      method = "lv"
    )
    
    best_idx <- which.min(dists)
    if (dists[best_idx] <= max_dist) {
      return(names_dict[best_idx])
    }
  }
  
  NA_character_
}

# Header and body parser
split_first_page_from_text <- function(full_text) {
  
  lines <- str_split(full_text, "\n")[[1]]
  lines <- str_trim(lines)
  lines <- lines[lines != ""]
  
  blacklist <- c(
    "^European Commission\\s*-?\\s*Speech$",
    "^Speech$",
    "^\\[.*Delivery\\]$"
  )
  
  loc_date_pattern <- "^[A-Za-zÀ-ÖØ-öø-ÿ .'-]+,\\s*(\\d{1,2} [A-Za-z]+ \\d{4}|
  [A-Za-z]+ \\d{1,2},\\d{4})"
  
  loc_idx <- which(str_detect(lines, loc_date_pattern))
  
  if (length(loc_idx) > 0) {
    
    idx <- loc_idx[1]
    loc_line <- lines[idx]
    
    heading_candidates <- lines[1:(idx - 1)]
    heading_candidates <- heading_candidates[
      !str_detect(heading_candidates, paste(blacklist, collapse = "|"))
    ]
    
    heading_full <- str_squish(paste(heading_candidates, collapse = " "))
    body <- paste(lines[(idx + 1):length(lines)], collapse = "\n")
    
  } else {
    heading_full <- NA_character_
    loc_line <- NA_character_
    body <- paste(lines, collapse = "\n")
  }
  
  list(
    heading_full = heading_full,
    loc_line     = loc_line,
    text_body    = body
  )
}

# Single extractor
extract_place <- function(loc_line) {
  if (is.na(loc_line)) return(NA_character_)
  m <- str_match(loc_line, "^([^,]+),")
  if (!is.na(m[, 2])) str_trim(m[, 2]) else NA_character_
}

extract_date_from_locline <- function(loc_line) {
  if (is.na(loc_line)) return(NA_character_)
  m <- str_match(loc_line, "(\\d{1,2} [A-Za-z]+ \\d{4}|[A-Za-z]+ \\d{1,2}, 
                 \\d{4})")
  if (!is.na(m[, 2])) m[, 2] else NA_character_
}

extract_event_context <- function(heading_full) {
  if (is.na(heading_full)) return(NA_character_)
  m <- str_match(heading_full, "\\b(on|at|for|following|ahead of)\\b\\s+(.+)$")
  if (!all(is.na(m))) m[, 3] else NA_character_
}

extract_speech_code <- function(text) {
  
  m <- str_match(text, "(SPEECH/\\d{2}/\\d{1,4})")
  
  if (!all(is.na(m))) {
    list(
      clean_text  = str_squish(str_replace(text, m[, 2], "")),
      speech_code = m[, 2]
    )
  } else {
    list(
      clean_text  = str_squish(text),
      speech_code = NA_character_
    )
  }
}

# Document extraction (pdf and txt (for translations))
extract_doc_info_generic <- function(path) {
  
  message("Process: ", basename(path))
  ext <- tolower(tools::file_ext(path))
  
  if (ext == "pdf") {
    pages <- pdf_text(path)
    full_text <- paste(pages, collapse = "\n")
    meta <- pdf_info(path)
  } else {
    full_text <- read_file(path)
    meta <- list(
      title = NA, author = NA,
      created = NA, modified = NA,
      pages = NA
    )
  }
  
  s <- split_first_page_from_text(full_text)
  
  place <- extract_place(s$loc_line)
  date  <- extract_date_from_locline(s$loc_line)
  
  speech <- extract_speech_code(s$text_body)
  
  speaker <- extract_speaker_dict(
    s$heading_full,
    names_dict,
    fuzzy = TRUE,
    max_dist = 1
  )
  
  event <- extract_event_context(s$heading_full)
  
  tibble(
    file          = basename(path),
    path          = path,
    heading_full  = s$heading_full,
    loc_line      = s$loc_line,
    place         = place,
    date          = date,
    speaker_name  = speaker,
    event_context = event,
    speech_code   = speech$speech_code,
    title_meta    = meta$title,
    author_meta   = meta$author,
    created       = meta$created,
    modified      = meta$modified,
    pages         = meta$pages,
    text          = speech$clean_text
  )
}

# Folders
build_corpus_for_dir <- function(dir_path) {
  
  files <- list.files(
    dir_path,
    pattern = "\\.(pdf|txt)$",
    full.names = TRUE,
    ignore.case = TRUE
  )
  
  if (length(files) == 0) {
    stop("No files in: ", dir_path)
  }
  
  map_dfr(files, ~ tryCatch(
    extract_doc_info_generic(.x),
    error = function(e) {
      message("Error in: ", basename(.x))
      NULL
    }
  ))
}

# Building the corpus and saving
corpus_25    <- build_corpus_for_dir(dir_25)
corpus_24    <- build_corpus_for_dir(dir_24)
corpus_trans <- build_corpus_for_dir(dir_trans)

write_csv(corpus_25, csv_25)
write_csv(corpus_24, csv_24)
write_csv(corpus_trans, csv_trans)

corpus_all <- bind_rows(corpus_24, corpus_25, corpus_trans)
write_csv(corpus_all, "corpus_eu_speeches_24_25_trans.csv")

# 2 Cleaning the Data 

d <- read.csv("./corpus_eu_speeches_24_25_trans.csv")

## Speaker
d$speaker_name <- gsub("VÁRHELYI", "Várhelyi", d$speaker_name)
d$speaker_name <- gsub("Dombrovska", "Dombrovskis", d$speaker_name)
d$speaker_name <- gsub("Tzitikostas", "Tzitzikostas", d$speaker_name)

usn <- unique(d$speaker_name)

## Header
d$heading_full <- sub(
  "^European Commission\\s*-\\s*",
  "",
  d$heading_full
)
d$heading_full <- sub(
  "^\\[[^]]+\\]\\s*",
  "",
  d$heading_full
)

## Dates
d$date <- as.Date(d$date, format = "%d %B %Y")

## Language
d$language <- detect_language(d$text)

d$non_english <- ifelse(d$language == "en", 0, 1)

# Keep speeches that are in English
d <- d[d$non_english == 0, ]

## Text
pattern <- regex(
  "[\\[\\(\"\\-–—]*\\s*check\\s+against\\s+delivery\\s*[\\]\\)\"\\-–—]*",
  ignore_case = TRUE
)

# Clean text
d$text <- str_remove_all(d$text, pattern)

# Remove unnecessary blank spaces
d$text <- str_squish(d$text)

## Clean
d[c("created", "modified", "pages", "title_meta", "author_meta")] <- NULL

sum(is.na(d))

## Export ##
write_csv(d, "corpus_eu_speeches_clean.csv")

clean <- read_excel("corpus_eu_speeches.xlsx")

## Column names
colnames(clean) <- clean[1, ]   # erste Zeile als Spaltennamen setzen
clean <- clean[-1, ]            # erste Zeile aus den Daten entfernen
rownames(clean) <- NULL      # Zeilennummern neu setzen

## New column
clean$translation <- ifelse(grepl("_t\\.txt$", clean$file), 1, 0)

write_csv(clean, "clean_data.csv")


# 3 Extra dataset (sentence-level)


df <- read.csv("./clean_data.csv")

# Calculate the number of sentences and words per speech
df_counts <- df %>%
  group_by(file) %>%
  summarise(
    sentences  = str_count(first(text), "[.!?]"),
    words      = str_count(first(text), "\\S+"),
    heading_full   = first(heading_full),
    place          = first(place),
    date           = first(date),
    speaker_name   = first(speaker_name),
    event_context  = first(event_context),
    language       = first(language),
    non_english    = first(non_english),
    translation    = first(translation),
    .groups = "drop"
  )

# Divide text into individual sentences
df_saetze <- df %>%
  mutate(
    satz_liste = str_split(text, "(?<=[.!?])\\s+")
  ) %>%
  unnest(satz_liste) %>%
  group_by(file) %>%
  mutate(
    satz_nr = row_number()
  ) %>%
  ungroup() %>%
  rename(
    satz_text = satz_liste
  )

# Merge sentence data with speech statistics
df_final <- df_saetze %>%
  left_join(df_counts, by = "file") %>%
  select(
    file,
    satz_nr,
    satz_text,
    sentences,
    words,
  )

df_extra <- df_final %>%
  left_join(df, by = "file") %>%
  select(
    file,
    satz_nr,
    satz_text,
    sentences,
    words,
    heading_full,
    place,
    date,
    speaker_name,
    event_context,
    language,
    non_english,
    translation
  )

write_csv(df_extra, "df_sentences.csv")  


# 4 Graphics


df <- read_excel("bertopic.xlsx")

df <- df %>%
  mutate(
    date = as.Date(date),
    month = floor_date(date, "month")
  )

reden_gesamt <- df %>%
  distinct(file, month) %>%
  count(month, name = "anzahl")

reden_defence <- df %>%
  filter(is_european_defence == "True") %>%
  distinct(file, month) %>%
  count(month, name = "anzahl")

saetze_defence <- df %>%
  filter(is_european_defence == "True") %>%
  count(month, name = "anzahl")

saetze_gesamt <- df %>%
  count(month, name = "anzahl")

plot_data <- bind_rows(
  reden_gesamt %>% mutate(typ = "Reden gesamt"),
  reden_defence %>% mutate(typ = "Reden Verteidigung"),
  saetze_defence %>% mutate(typ = "Sätze Verteidigung"),
  saetze_gesamt %>% mutate(typ = "Sätze gesamt")
)

plot_data <- plot_data %>%
  mutate(gruppe = ifelse(grepl("Reden", typ), "Reden", "Sätze"))

# Combining graphics
g1 <- ggplot() +
  # Beams: all Reason
  geom_col(data = reden_gesamt,
           aes(x = month, y = anzahl),
           fill = "grey80") +
  
  # Line: Speeches Defense
  geom_line(data = reden_defence,
            aes(x = month, y = anzahl, group = 1),
            color = "black") +
  
  # Points: Speeches Defense
  geom_point(data = reden_defence,
             aes(x = month, y = anzahl),
             color = "black",
             size = 2) +
  
  scale_x_date(date_labels = "%Y-%m") +
  labs(
    title = "Reden pro Monat und Reden zu Verteidigung",
    x = "Monat",
    y = "Anzahl Reden"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

g2 <- ggplot() +
  # Bar: all sentences
  geom_col(data = saetze_gesamt,
           aes(x = month, y = anzahl),
           fill = "grey80") +
  
  # Line: Sentences Defense
  geom_line(data = saetze_defence,
            aes(x = month, y = anzahl, group = 1),
            color = "black") +
  
  # Punkte: Sentences Defense
  geom_point(data = saetze_defence,
             aes(x = month, y = anzahl),
             color = "black",
             size = 2) +
  
  scale_x_date(date_labels = "%Y-%m") +
  labs(
    title = "Sätze pro Monat und Sätze zu Verteidigung",
    x = "Monat",
    y = "Anzahl Sätze"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

g1 / g2


# 5 Evaluation Classifier


df <- read.csv("./df_with_bertopic_sentence_and_context.csv")
write_xlsx(df, "bertopic.xlsx")

# Definition weight variable
weight_var <- "is_european_defence"

# Generate a random sample of 50% from the DataFrame with weighting of the 
# variable "weight_var"
sample <- df %>%
  group_by(is_european_defence) %>%
  slice_sample(n = 250) %>%
  ungroup()

write_xlsx(sample, "bertopic_sample.xlsx")

df_sample <- read_excel("bertopic_sample.xlsx")

# Calculating F1-score
is.numeric(df_sample$right_is_defence)
df_sample$right_is_defence <- as.logical(df_sample$right_is_defence)

df_sample$is_european_defence[df_sample$is_european_defence == "True"] <- 1
df_sample$is_european_defence[df_sample$is_european_defence == "False"] <- 0

class(df_sample$is_european_defence)
df_sample$is_european_defence <- as.numeric(df_sample$is_european_defence)
df_sample$is_european_defence <- as.logical(df_sample$is_european_defence)

df_sample$confusion <- with(df_sample,
                            ifelse(right_is_defence == TRUE  & 
                                     is_european_defence == TRUE,  "TP",
                                   ifelse(right_is_defence == FALSE & 
                                            is_european_defence == TRUE,  "FP",
                                          ifelse(right_is_defence == FALSE & 
                                                   is_european_defence == FALSE, 
                                                 "TN",
                                                 ifelse(right_is_defence == TRUE  
                                                        & is_european_defence == 
                                                          FALSE, 
                                                        "FN", NA))))
)

precision <- sum(df_sample$confusion == "TP", na.rm = TRUE) / 
  (sum(df_sample$confusion == "TP", na.rm = TRUE) + 
     sum(df_sample$confusion == "FP", na.rm = TRUE))

recall <- sum(df_sample$confusion == "TP", na.rm = TRUE) / 
  (sum(df_sample$confusion == "TP", na.rm = TRUE) + 
     sum(df_sample$confusion == "FN", na.rm = TRUE))

accuracy <- (sum(df_sample$confusion == "TP", na.rm = TRUE) + 
               sum(df_sample$confusion == "TN", na.rm = TRUE)) / 
  (sum(df_sample$confusion == "TP", na.rm = TRUE) + 
     sum(df_sample$confusion == "FP", na.rm = TRUE) + 
     sum(df_sample$confusion == "FN", na.rm = TRUE) + 
     sum(df_sample$confusion == "TN", na.rm = TRUE))

f1 <- (2 * precision * recall) / (precision + recall)

## Load dataset
df1 <- read.csv("./topics_info.csv")
write_xlsx(df1, "topics.xlsx")


# 6 Intra-coder reliability

# Load dataset
df2 <- read_excel("./bertopic_sample_2.xlsx")

# Cohens Kappa
kappa2(df2[, c("right_is_defence", "still_is_defence")])

