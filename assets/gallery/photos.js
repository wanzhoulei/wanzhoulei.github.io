// Photos shown on gallery.html. Each photo goes in the section named by "region" (china, neotropics,
// or other), in the order listed here. "featured": true makes a large 2x2 tile.
//
// Easiest: python3 tools/add_photos.py PHOTO.jpg --region china --species "Aldania imitans" --cn "仿斑伞蛱蝶" \
//              --place "Mingyong, Yunnan" --date "Jun 2026"
// That resizes the photo, strips its metadata (including GPS), and adds the entry for you.
//
// By hand: put the image in assets/gallery/ (and a smaller copy with the same name in assets/gallery/thumbs/),
// then add one line. Only "file" and "region" are required; keep the list valid JSON (double quotes,
// a comma between entries, none after the last).
window.GALLERY = [
  {"file": "aldania-imitans.jpg", "w": 2000, "h": 1334, "region": "china", "species": "Aldania imitans", "cn": "仿斑伞蛱蝶", "place": "Mingyong, Yunnan", "date": "Jun 2026", "featured": true},
  {"file": "lexias-cyanipardus.jpg", "w": 2000, "h": 1334, "region": "china", "species": "Lexias cyanipardus", "cn": "蓝豹律蛱蝶", "date": "Apr 2026"},
  {"file": "anthocharis-bieti.jpg", "w": 2000, "h": 1333, "region": "china", "species": "Anthocharis bieti", "cn": "皮氏尖襟粉蝶", "place": "Pudacuo, Yunnan", "date": "Jun 2026"},
  {"file": "abrota-ganga.jpg", "w": 2000, "h": 1333, "region": "china", "species": "Abrota ganga", "cn": "婀蛱蝶", "place": "Dalonggang", "date": "Jun 2024"},
  {"file": "davidina-armandi.jpg", "w": 2000, "h": 1333, "region": "china", "species": "Davidina armandi", "cn": "绢眼蝶", "place": "Beijing", "date": "May 2026"},
  {"file": "dichorragia-nesseus.jpg", "w": 2000, "h": 1334, "region": "china", "species": "Dichorragia nesseus", "cn": "长波电蛱蝶", "place": "Mingyong, Yunnan", "date": "Jun 2026"},
  {"file": "thauria-lathyi.jpg", "w": 2000, "h": 1333, "region": "china", "species": "Thauria lathyi", "cn": "斜带环蝶", "date": "Apr 2026"},
  {"file": "beixienigonghuidie.jpg", "w": 2000, "h": 1333, "region": "china", "cn": "北协拟工灰蝶", "date": "May 2026"},
  {"file": "bocuijiadie.jpg", "w": 2000, "h": 1333, "region": "china", "cn": "铂翠蛱蝶", "place": "Dalonggang", "date": "Jun 2024"},
  {"file": "araragi-enthea.jpg", "w": 2000, "h": 1333, "region": "china", "species": "Araragi enthea", "cn": "癞灰蝶", "date": "Jun 2026"},
  {"file": "lethe-gracilis.jpg", "w": 2000, "h": 1333, "region": "china", "species": "Lethe gracilis", "cn": "纤细黛眼蝶", "place": "Mingyong, Yunnan", "date": "Jun 2026"},
  {"file": "dongfangcuijiadie.jpg", "w": 2000, "h": 1334, "region": "china", "cn": "东方翠蛱蝶", "date": "Jul 2026"},
  {"file": "acropolis-thalia-male.jpg", "w": 2000, "h": 1333, "region": "china", "species": "Acropolis thalia", "cn": "颠眼蝶", "note": "male", "place": "Dalonggang", "date": "Jun 2024"},
  {"file": "pseudohaetera-hypaesia-underside.jpg", "w": 2000, "h": 1334, "region": "neotropics", "species": "Pseudohaetera hypaesia", "note": "underside", "place": "Santa María", "date": "Aug 2022", "featured": true},
  {"file": "callicorina-pulchra.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Callicorina pulchra", "date": "Nov 2025"},
  {"file": "perisama-dorbignyi.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Perisama dorbignyi", "date": "Nov 2025"},
  {"file": "morpho-rhodopteron.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Morpho rhodopteron", "date": "Aug 2022"},
  {"file": "rhamma-familiaris.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Rhamma familiaris", "place": "Huayna Picchu, Peru", "date": "May 2024"},
  {"file": "archaeoprepona-licomedes-upperside.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Archaeoprepona licomedes", "note": "upperside", "date": "Nov 2025", "featured": true},
  {"file": "perisama-dorbignyi-upperside.jpg", "w": 2000, "h": 1334, "region": "neotropics", "species": "Perisama dorbignyi", "note": "upperside", "date": "Nov 2025"},
  {"file": "lymanopoda-nevada.jpg", "w": 2000, "h": 1334, "region": "neotropics", "species": "Lymanopoda nevada", "place": "Sierra Nevada de Santa Marta, Colombia", "date": "Aug 2022"},
  {"file": "archaeoprepona-licomedes.jpg", "w": 2000, "h": 1334, "region": "neotropics", "species": "Archaeoprepona licomedes", "date": "Nov 2025"},
  {"file": "perisama-camelita-underside.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Perisama camelita", "note": "underside", "date": "Nov 2025"},
  {"file": "neographium-agesilaus.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Neographium agesilaus", "date": "Mar 2024", "featured": true},
  {"file": "photo-2024-03-16.jpg", "w": 2000, "h": 1333, "region": "neotropics", "date": "Mar 2024"},
  {"file": "morpho-zephyritis.jpg", "w": 2000, "h": 1334, "region": "neotropics", "species": "Morpho zephyritis", "date": "Nov 2025"},
  {"file": "junea-dorinda.jpg", "w": 2000, "h": 1333, "region": "neotropics", "species": "Junea dorinda", "date": "Nov 2025"},
  {"file": "peru-2024-05.jpg", "w": 2000, "h": 1333, "region": "neotropics", "place": "Peru", "date": "May 2024"}
];
