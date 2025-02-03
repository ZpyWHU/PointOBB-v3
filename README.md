<p align="center">
  <h1 align="center">PointOBB-v3： Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection</h1>
  <p align="center">
      <a href='https://scholar.google.com.hk/citations?user=rQbW67AAAAAJ&hl' style='text-decoration: none' >Peiyuan Zhang</a><sup></sup>&emsp;
      <a href='https://scholar.google.com.hk/citations?hl=zh-CN&user=6XibZaYAAAAJ' style='text-decoration: none' >Junwei Luo</a><sup></sup>&emsp;
      <a href='https://yangxue0827.github.io/' style='text-decoration: none' >Xue Yang</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=OYtSc4AAAAAJ&hl=en' style='text-decoration: none' >Yi Yu</a><sup></sup>&emsp; 
      <a href='https://scholar.google.com/citations?hl=en&user=TvsTun4AAAAJ' style='text-decoration: none' >Qingyun Li</a><sup></sup>&emsp;   
      <a href='https://scholar.google.com.hk/citations?user=v-aQ8GsAAAAJ&hl=zh-CN' style='text-decoration: none' >Yue Zhou</a><sup></sup>&emsp;
      <a href='https://jiaxiaosong1002.github.io/' style='text-decoration: none' >Xiaosong Jia</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=G9jWIggAAAAJ&hl=en' style='text-decoration: none' >Xudong Lu</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=8SCEv-YAAAAJ&hl=en' style='text-decoration: none' >Jingdong Chen</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=oamjJdYAAAAJ&hl=zh-CN' style='text-decoration: none' >Xiang Li</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=ga230VoAAAAJ&hl=en' style='text-decoration: none' >Junchi Yan</a><sup></sup>&emsp;
      <a href='https://scholar.google.com/citations?user=wn9hc6UAAAAJ&hl=zh-CN' style='text-decoration: none' >Yansheng Li</a><sup></sup>&emsp;      
      <div align="center">
      <a href='https://arxiv.org/abs/2501.13898'><img src='https://img.shields.io/badge/arXiv-2501.09720-brown.svg?logo=arxiv&logoColor=white'></a>
	  </div>
    <p align='center'>
        If you find our work helpful, please consider giving us a ⭐!
    </p>
   </p>
</p>

This repo hosts the official implementation of the paper: **PointOBB-v3: Expanding Performance Boundaries of Single Point-Supervised Oriented Object Detection**.

---
## Abstract
  With the growing demand for oriented object detection (OOD), recent studies on point-supervised OOD have attracted significant interest. In this paper, we propose PointOBB-v3, a stronger single point-supervised OOD framework. Compared to existing methods, it generates pseudo rotated boxes without additional priors and incorporates support for the end-to-end paradigm. PointOBB-v3 functions by integrating three unique image views: the original view, a resized view, and a rotated/flipped (rot/flp) view. Based on the views, a scale augmentation module and an angle acquisition module are constructed. In the first module, a Scale-Sensitive Consistency (SSC) loss and a Scale-Sensitive Feature Fusion (SSFF) module are introduced to improve the model's ability to estimate object scale. To achieve precise angle predictions, the second module employs symmetry-based self-supervised learning. Additionally, we introduce an end-to-end version that eliminates the pseudo-label generation process by integrating a detector branch and introduces an Instance-Aware Weighting (IAW) strategy to focus on high-quality predictions. We conducted extensive experiments on the DIOR-R, DOTA-v1.0/v1.5/v2.0, FAIR1M, STAR, and RSAR datasets. Across all these datasets, our method achieves an average improvement in accuracy of 3.56\% in comparison to previous state-of-the-art methods.

---
## Guide
If you need the `two_stage` version, please switch to the `two_stage` branch to get the code.  
If you need the `end_to_end` version, please switch to the `end_to_end` branch to get the code.
