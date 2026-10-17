// GPT-as-Policy baselines retain their published values and aggregation.
// RoboICL: ten-task panel scores and separately marked reconstructed SR estimates.
// GPT-Policy: owner-confirmed paper Table 2. See assets/benchmarks/README.md.
const benchmarkData = {
  "version": "2026-10-12",
  "paper": "https://arxiv.org/abs/2609.19138",
  "paperCompiledAt": "2026-10-11T14:51:05-07:00",
  "paperSha256": "7f423f44d4d4fcdaa690adc09d41b9af8aa5d91f96a9d3117953d8f3d7e09134",
  "reference": "https://anonymous-report-421.github.io/public-website/?view=1",
  "models": [
    {
      "id": "DM0.5",
      "name": "DM0.5",
      "header": "DM0.5",
      "score": 31.8206,
      "success": 22.0404,
      "logos": [
        "assets/benchmarks/logos/dm.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "GalaxeaVLA (G0.5)",
      "name": "Galaxea G0.5",
      "header": "GalaxeaVLA (G0.5)",
      "score": 38.2608,
      "success": 30.9464,
      "logos": [
        "assets/benchmarks/logos/galaxea.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "Xiaomi-Robotics-1",
      "name": "Xiaomi R1",
      "header": "Xiaomi-Robotics-1",
      "score": 33.9746,
      "success": 25.5072,
      "logos": [
        "assets/benchmarks/logos/xiaomi.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "OpenWAM-α",
      "name": "OpenWAM-α",
      "header": "OpenWAM-α",
      "score": 32.7322,
      "success": 23.0662,
      "logos": [
        "assets/benchmarks/logos/openwam.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "Meituan-Robotics-0",
      "name": "Meituan R0",
      "header": "Meituan-Robotics-0",
      "score": 28.328599999999998,
      "success": 17.8796,
      "logos": [
        "assets/benchmarks/logos/meituan.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "Hy-Embodied-0.5-VLA",
      "name": "Hy-VLA 0.5",
      "header": "Hy-Embodied-0.5-VLA",
      "score": 21.1594,
      "success": 13.935,
      "logos": [
        "assets/benchmarks/logos/hunyuan.svg"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "Spatial Forcing",
      "name": "Spatial Forcing",
      "header": "Spatial Forcing",
      "score": 24.9644,
      "success": 17.1458,
      "logos": [
        "assets/benchmarks/logos/spatial.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "Pi-05",
      "name": "π₀.₅",
      "header": "π₀.₅",
      "score": 24.43,
      "success": 15.6666,
      "logos": [
        "assets/benchmarks/logos/pi.png"
      ],
      "kind": "pi",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "InternVLA-A1.5",
      "name": "InternVLA A1.5",
      "header": "InternVLA-A1.5",
      "score": 21.7398,
      "success": 15.0268,
      "logos": [
        "assets/benchmarks/logos/internvla.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "StarVLA-PI_v3",
      "name": "StarVLA-PI",
      "header": "StarVLA-PI_v3",
      "score": 22.3636,
      "success": 16.1734,
      "logos": [
        "assets/benchmarks/logos/starvla.png"
      ],
      "kind": "reference",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "mix",
      "name": "π₀.₅ + GPT 6 Astra",
      "header": "π₀.₅ + GPT 6 Astra",
      "score": 62.6,
      "success": 48,
      "logos": [
        "assets/benchmarks/logos/pi.png",
        "assets/benchmarks/logos/openai.ico"
      ],
      "kind": "hybrid",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "gpt",
      "name": "GPT 6 Astra (direct)",
      "header": "GPT 6 Astra (direct)",
      "score": 37.8125,
      "success": 26,
      "logos": [
        "assets/benchmarks/logos/openai.ico"
      ],
      "kind": "direct",
      "source": "gpt-as-policy-report"
    },
    {
      "id": "roboicl",
      "name": "RoboICL (1-shot)",
      "barLabel": "RoboICL",
      "setting": "1-shot",
      "header": "RoboICL<br>1-shot",
      "score": 60.6,
      "success": null,
      "logos": [
        "assets/benchmarks/logos/openai.ico"
      ],
      "kind": "roboicl",
      "source": "roboicl-ten-task-panel"
    },
    {
      "id": "ours",
      "name": "GPT-Policy",
      "setting": "1-shot",
      "header": "GPT-Policy<br>1-shot (Ours)",
      "score": 80.2,
      "success": 72,
      "logos": [
        "assets/morphi-logo.png",
        "assets/benchmarks/logos/openai.ico"
      ],
      "kind": "ours",
      "source": "confirmed-paper-table-2"
    }
  ],
  "tasks": [
    {
      "id": "organize_table",
      "en": "Organize table",
      "zh": "整理桌面",
      "successes": 1,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 44.0,
          "success": 4.0
        },
        "GalaxeaVLA (G0.5)": {
          "score": 46.33,
          "success": 11.33
        },
        "Xiaomi-Robotics-1": {
          "score": 57.67,
          "success": 14.669999999999998
        },
        "OpenWAM-α": {
          "score": 62.5,
          "success": 13.330000000000002
        },
        "Meituan-Robotics-0": {
          "score": 43.67,
          "success": 4.0
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 42.0,
          "success": 4.67
        },
        "Spatial Forcing": {
          "score": 31.670000000000005,
          "success": 5.33
        },
        "Pi-05": {
          "score": 23.33,
          "success": 0.0
        },
        "InternVLA-A1.5": {
          "score": 35.67,
          "success": 2.67
        },
        "StarVLA-PI_v3": {
          "score": 25.17,
          "success": 2.0
        },
        "mix": {
          "score": 60.0,
          "success": 0
        },
        "gpt": {
          "score": 30.0,
          "success": 0
        },
        "ours": {
          "score": 65,
          "success": 20
        },
        "roboicl": {
          "score": 50,
          "success": 0,
          "successEstimated": true,
          "zeroShot": false
        }
      }
    },
    {
      "id": "classify_objects_by_language",
      "en": "Classify by language",
      "zh": "语言分类",
      "successes": 4,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 0.4699999999999999,
          "success": 0.0
        },
        "GalaxeaVLA (G0.5)": {
          "score": 1.07,
          "success": 0.0
        },
        "Xiaomi-Robotics-1": {
          "score": 2.0,
          "success": 0.67
        },
        "OpenWAM-α": {
          "score": 1.33,
          "success": 0.0
        },
        "Meituan-Robotics-0": {
          "score": 2.47,
          "success": 0.67
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 0.53,
          "success": 0.0
        },
        "Spatial Forcing": {
          "score": 0.6,
          "success": 0.0
        },
        "Pi-05": {
          "score": 0.6,
          "success": 0.0
        },
        "InternVLA-A1.5": {
          "score": 0.13,
          "success": 0.0
        },
        "StarVLA-PI_v3": {
          "score": 0.27,
          "success": 0.0
        },
        "mix": {
          "score": 38.0,
          "success": 20
        },
        "gpt": {
          "score": 60.0,
          "success": 40
        },
        "ours": {
          "score": 88,
          "success": 80
        },
        "roboicl": {
          "score": 44,
          "success": 40,
          "successEstimated": true,
          "zeroShot": true
        }
      }
    },
    {
      "id": "imitate_sorting_sequence",
      "en": "Imitate sorting sequence",
      "zh": "顺序模仿",
      "successes": 5,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 1.8,
          "success": 0.0
        },
        "GalaxeaVLA (G0.5)": {
          "score": 1.67,
          "success": 0.0
        },
        "Xiaomi-Robotics-1": {
          "score": 2.5,
          "success": 0.67
        },
        "OpenWAM-α": {
          "score": 2.9,
          "success": 0.67
        },
        "Meituan-Robotics-0": {
          "score": 1.53,
          "success": 0.0
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 0.9,
          "success": 0.0
        },
        "Spatial Forcing": {
          "score": 1.3,
          "success": 0.0
        },
        "Pi-05": {
          "score": 1.6,
          "success": 0.0
        },
        "InternVLA-A1.5": {
          "score": 0.6,
          "success": 0.0
        },
        "StarVLA-PI_v3": {
          "score": 0.73,
          "success": 0.0
        },
        "mix": {
          "score": 53.0,
          "success": 40
        },
        "gpt": {
          "score": 0.0,
          "success": 0
        },
        "ours": {
          "score": 100,
          "success": 100
        },
        "roboicl": {
          "score": 90,
          "success": 80,
          "successEstimated": true,
          "zeroShot": false
        }
      }
    },
    {
      "id": "arrange_largest_number",
      "en": "Arrange largest number",
      "zh": "排列最大数",
      "successes": 5,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 7.853999999999999,
          "success": 3.2
        },
        "GalaxeaVLA (G0.5)": {
          "score": 4.106,
          "success": 0.532
        },
        "Xiaomi-Robotics-1": {
          "score": 8.562,
          "success": 3.732
        },
        "OpenWAM-α": {
          "score": 4.362,
          "success": 0.532
        },
        "Meituan-Robotics-0": {
          "score": 6.414000000000001,
          "success": 2.668
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 0.852,
          "success": 0.0
        },
        "Spatial Forcing": {
          "score": 2.158,
          "success": 0.532
        },
        "Pi-05": {
          "score": 2.2920000000000003,
          "success": 0.532
        },
        "InternVLA-A1.5": {
          "score": 0.5740000000000001,
          "success": 0.0
        },
        "StarVLA-PI_v3": {
          "score": 1.762,
          "success": 0.0
        },
        "mix": {
          "score": 50.0,
          "success": 40
        },
        "gpt": {
          "score": 57.0,
          "success": 40
        },
        "ours": {
          "score": 100,
          "success": 100
        },
        "roboicl": {
          "score": 58,
          "success": null,
          "successEstimated": false,
          "zeroShot": false
        }
      }
    },
    {
      "id": "build_tower",
      "en": "Build tower",
      "zh": "搭塔",
      "successes": 2,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 55.2,
          "success": 42.0
        },
        "GalaxeaVLA (G0.5)": {
          "score": 82.93,
          "success": 78.67
        },
        "Xiaomi-Robotics-1": {
          "score": 52.6,
          "success": 47.33
        },
        "OpenWAM-α": {
          "score": 52.529999999999994,
          "success": 38.67
        },
        "Meituan-Robotics-0": {
          "score": 32.73,
          "success": 11.33
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 20.87,
          "success": 12.0
        },
        "Spatial Forcing": {
          "score": 43.93,
          "success": 32.67
        },
        "Pi-05": {
          "score": 37.73,
          "success": 24.0
        },
        "InternVLA-A1.5": {
          "score": 37.4,
          "success": 28.670000000000005
        },
        "StarVLA-PI_v3": {
          "score": 63.2,
          "success": 55.33
        },
        "mix": {
          "score": 64.0,
          "success": 60
        },
        "gpt": {
          "score": 12.0,
          "success": 0
        },
        "ours": {
          "score": 46,
          "success": 40
        },
        "roboicl": {
          "score": 100,
          "success": 100,
          "successEstimated": true,
          "zeroShot": false
        }
      }
    },
    {
      "id": "fold_clothes",
      "en": "Fold clothes",
      "zh": "折衣服",
      "successes": 4,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 28.962,
          "success": 24.534
        },
        "GalaxeaVLA (G0.5)": {
          "score": 32.75,
          "success": 28.002
        },
        "Xiaomi-Robotics-1": {
          "score": 41.49,
          "success": 38.932
        },
        "OpenWAM-α": {
          "score": 51.31,
          "success": 49.068
        },
        "Meituan-Robotics-0": {
          "score": 19.787999999999997,
          "success": 14.929999999999998
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 20.108,
          "success": 17.602
        },
        "Spatial Forcing": {
          "score": 36.37,
          "success": 31.729999999999997
        },
        "Pi-05": {
          "score": 29.118000000000002,
          "success": 21.066000000000003
        },
        "InternVLA-A1.5": {
          "score": 16.162000000000003,
          "success": 13.598
        },
        "StarVLA-PI_v3": {
          "score": 32.214,
          "success": 28.532
        },
        "mix": {
          "score": 100.0,
          "success": 100
        },
        "gpt": {
          "score": 40.0,
          "success": 40
        },
        "ours": {
          "score": 80,
          "success": 80
        },
        "roboicl": {
          "score": 84,
          "success": null,
          "successEstimated": false,
          "zeroShot": false
        }
      }
    },
    {
      "id": "put_bottles_into_dustbin",
      "en": "Put bottles into dustbin",
      "zh": "瓶子入垃圾桶",
      "successes": 3,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 81.7,
          "success": 72.0
        },
        "GalaxeaVLA (G0.5)": {
          "score": 96.3,
          "success": 94.0
        },
        "Xiaomi-Robotics-1": {
          "score": 97.7,
          "success": 96.67
        },
        "OpenWAM-α": {
          "score": 94.03,
          "success": 91.33
        },
        "Meituan-Robotics-0": {
          "score": 82.5,
          "success": 72.67
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 90.0,
          "success": 84.67
        },
        "Spatial Forcing": {
          "score": 85.6,
          "success": 78.0
        },
        "Pi-05": {
          "score": 79.93,
          "success": 69.33
        },
        "InternVLA-A1.5": {
          "score": 86.23,
          "success": 79.33
        },
        "StarVLA-PI_v3": {
          "score": 68.67,
          "success": 56.67
        },
        "mix": {
          "score": 100.0,
          "success": 100
        },
        "gpt": {
          "score": 36.0,
          "success": 20
        },
        "ours": {
          "score": 73,
          "success": 60
        },
        "roboicl": {
          "score": 73,
          "success": 60,
          "successEstimated": true,
          "zeroShot": false
        }
      }
    },
    {
      "id": "pack_objects_into_box",
      "en": "Pack objects into box",
      "zh": "物品装箱",
      "successes": 3,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 14.719999999999999,
          "success": 1.33
        },
        "GalaxeaVLA (G0.5)": {
          "score": 17.122,
          "success": 2.93
        },
        "Xiaomi-Robotics-1": {
          "score": 18.694,
          "success": 2.398
        },
        "OpenWAM-α": {
          "score": 20.830000000000002,
          "success": 3.732
        },
        "Meituan-Robotics-0": {
          "score": 13.614,
          "success": 3.198
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 12.533999999999999,
          "success": 1.068
        },
        "Spatial Forcing": {
          "score": 14.346,
          "success": 1.866
        },
        "Pi-05": {
          "score": 18.36,
          "success": 2.398
        },
        "InternVLA-A1.5": {
          "score": 8.931999999999999,
          "success": 0.0
        },
        "StarVLA-PI_v3": {
          "score": 7.69,
          "success": 0.532
        },
        "mix": {
          "score": 50.0,
          "success": 20
        },
        "gpt": {
          "score": 50.0,
          "success": 20
        },
        "ours": {
          "score": 70,
          "success": 60
        },
        "roboicl": {
          "score": 36,
          "success": null,
          "successEstimated": false,
          "zeroShot": false
        }
      }
    },
    {
      "id": "make_kong",
      "en": "Make Kong",
      "zh": "Make Kong",
      "successes": 4,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 56.67,
          "success": 56.67
        },
        "GalaxeaVLA (G0.5)": {
          "score": 90.0,
          "success": 90.0
        },
        "Xiaomi-Robotics-1": {
          "score": 41.33,
          "success": 41.33
        },
        "OpenWAM-α": {
          "score": 32.0,
          "success": 32.0
        },
        "Meituan-Robotics-0": {
          "score": 56.0,
          "success": 56.0
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 18.67,
          "success": 18.67
        },
        "Spatial Forcing": {
          "score": 12.0,
          "success": 12.0
        },
        "Pi-05": {
          "score": 26.670000000000005,
          "success": 26.670000000000005
        },
        "InternVLA-A1.5": {
          "score": 26.0,
          "success": 26.0
        },
        "StarVLA-PI_v3": {
          "score": 18.0,
          "success": 18.0
        },
        "mix": {
          "score": 40.0,
          "success": 40
        },
        "gpt": {
          "score": 0.0,
          "success": 0
        },
        "ours": {
          "score": 80,
          "success": 80
        },
        "roboicl": {
          "score": 0,
          "success": 0,
          "successEstimated": true,
          "zeroShot": false
        }
      }
    },
    {
      "id": "classify_objects",
      "en": "Classify objects",
      "zh": "物品分类",
      "successes": 5,
      "trials": 5,
      "values": {
        "DM0.5": {
          "score": 26.829999999999995,
          "success": 16.67
        },
        "GalaxeaVLA (G0.5)": {
          "score": 10.33,
          "success": 4.0
        },
        "Xiaomi-Robotics-1": {
          "score": 17.2,
          "success": 8.67
        },
        "OpenWAM-α": {
          "score": 5.53,
          "success": 1.33
        },
        "Meituan-Robotics-0": {
          "score": 24.57,
          "success": 13.330000000000002
        },
        "Hy-Embodied-0.5-VLA": {
          "score": 5.13,
          "success": 0.67
        },
        "Spatial Forcing": {
          "score": 21.67,
          "success": 9.33
        },
        "Pi-05": {
          "score": 24.67,
          "success": 12.67
        },
        "InternVLA-A1.5": {
          "score": 5.7,
          "success": 0.0
        },
        "StarVLA-PI_v3": {
          "score": 5.93,
          "success": 0.67
        },
        "mix": {
          "score": 71.0,
          "success": 60
        },
        "gpt": {
          "score": 100.0,
          "success": 100
        },
        "ours": {
          "score": 100,
          "success": 100
        },
        "roboicl": {
          "score": 71,
          "success": 60,
          "successEstimated": true,
          "zeroShot": false
        }
      }
    }
  ],
  "heatmapModels": [
    "DM0.5",
    "GalaxeaVLA (G0.5)",
    "Xiaomi-Robotics-1",
    "OpenWAM-α",
    "Pi-05",
    "mix",
    "gpt",
    "roboicl",
    "ours"
  ],
  "resultsSource": "User-confirmed Table 2 screenshot, 2026-10-11; supersedes simulation results in the local PDF export.",
  "heatmapTasks": [
    "organize_table",
    "classify_objects_by_language",
    "imitate_sorting_sequence",
    "arrange_largest_number",
    "pack_objects_into_box",
    "classify_objects",
    "build_tower",
    "make_kong",
    "fold_clothes",
    "put_bottles_into_dustbin"
  ],
  "roboiclReference": "https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/#results-and-episodes"
};
