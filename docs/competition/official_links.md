# Official Links and Resources

## Challenge website (Google Site)
- Home: https://sites.google.com/view/abc2027-trackid3x3-challenge
- The Challenge: https://sites.google.com/view/abc2027-trackid3x3-challenge/the-challenge
- Registration (rules, eligibility, paper, prizes): https://sites.google.com/view/abc2027-trackid3x3-challenge/registration
- Dataset & Tutorial: https://sites.google.com/view/abc2027-trackid3x3-challenge/dataset-and-tutorial
- FAQ (content is in images; read it in a browser): https://sites.google.com/view/abc2027-trackid3x3-challenge/faq
- Organisers: https://sites.google.com/view/abc2027-trackid3x3-challenge/organisers

## Forms
- Team registration (**closes Oct 27**): https://docs.google.com/forms/d/e/1FAIpQLScl7duLqA5MUZMO4Lz-k4JNYty8IjSwBKWypIE63f9XBoNFag/viewform
- Results submission: https://forms.gle/rMwK4pHm1Xp2joV27

## Data (Google Drive: "TrackID3x3 Challenge 2026 (shared)")
- Root: https://drive.google.com/drive/folders/1xrqYSUP_EHdAbvdHX_0kBDlJYMYw1xXY
  - `Training/` (19-EfRRxhukjzLBFhgtQENB4Q6WcoT076): `court_registration.csv`, `games_roster.csv`
    - `vidoes/` (1lJGS6Amp0MWLnoNysH100JTJhyx_MHaM): IMG_0104 … IMG_0111 `.MOV`, 1.2–1.48 GB each
    - `full player annotations/` (19uDIK1Sp6-oZ7CzaFC78i-62fjzuJxZC)
      - `Image-space bounding boxes/` (1ONKvnUcfPG33mB9aYJBmj6aZMP_3QXvb): 8 txt
      - `court position/` (1JHXE8su46m9nRweCq0WonmPnNQhBbVwg): 8 txt
  - `Final challenge/` (1F6O91HPmkIYRoRh2uF_WuvYTFq6gFaso): IMG_0114.MP4, IMG_0115.MP4, `court_registration.csv` (corrected Oct 21), `games_roster.csv`
- Corrected challenge registration file: https://drive.google.com/file/d/1HOS2t-eodxaWYBUXG5R3KEpYMR52NYbP/view

Local status: all annotations, rosters and registrations are present in `data/`. **Training videos: only IMG_0106.MOV
is local; 7 are missing (~9.5 GB).**

## Tutorial
- Colab notebook: https://colab.research.google.com/drive/1I-ALe_yWJ9yEtsf6-NKgPYaHj_46AkV1
- YouTube: https://www.youtube.com/watch?v=z989J2gPc-E
- Zoom recording (password on the Dataset & Tutorial page): https://us02web.zoom.us/rec/share/jdAV0heJ2tZXZqnotDYlxFDvzapyEfzpNLOnbQxS70zpekscx4rU9bClZrB5ka-X.3ingym3KPHY7mDRb

## Paper and code
- Paper: https://arxiv.org/abs/2503.18282 (use **v2**, Aug 2025); DOI 10.1145/3728423.3759400
- Repo: https://github.com/open-starlab/TrackID3x3
  - Official metric: `script/compute_TI-HOTA.ipynb`. A copy of the code cells is in
    [../reference/compute_TI-HOTA_official.py](../reference/compute_TI-HOTA_official.py)
  - ⚠️ `ground_truth/Outdoor/` is **public annotation outside the official Training set, so it is prohibited**. Don't clone it into this project.

## Conference
- ABC 2027: https://autocare.ai/abc2027 (9th ABC, Jan 19–22, 2027, Sydney, hybrid)

## Contacts
- trackid3x3-challenge-2026@g.sp.m.is.nagoya-u.ac.jp (challenge-specific)
- abc2027@autocare.ai (subject line: "ABC Challenge")
- Organisers: Keisuke Fujii, Li Yin, Kazuhiro Yamada (STAL Lab, Nagoya University); Nazmun Nahid, Prerna Jamloki,
  Yuma Kosera, Sozo Inoue (Sozo Lab, Kyushu Institute of Technology)
