---
title: "Reducing Customer Churn by 15% via Behavioral Segmentation"
date: 2023-11-15
draft: false
tags: ["Strategy", "SQL", "Case Study"]
categories: ["Business Impact"]
summary: "How we identified at-risk users 3 weeks earlier than previous models."
---

## Executive Summary

At [Previous Company], our user retention dropped 5% in Q3. I led a squad to identify early warning signals of churn. We discovered that a drop in *session duration*, not *login frequency*, was the leading indicator.

## The Approach

We aggregated 50TB of event logs using BigQuery to reconstruct user journeys.

1.  **Hypothesis:** Users stop using specific features before they stop logging in.
2.  **Analysis:** Cohort analysis showed feature usage dropped 21 days before cancellation.
3.  **Action:** We implemented an automated email trigger when feature usage dipped by 10%.

## The Impact

| Metric | Before | After | Delta |
| :--- | :--- | :--- | :--- |
| Churn Rate | 4.2% | 3.5% | **-16%** |
| Win-back Rate | 2% | 8% | **+300%** |

## Technologies Used
- **Google BigQuery** for log processing
- **Tableau** for stakeholder dashboards
- **Python (Scikit-learn)** for the propensity model

## Conclusion

Data volume wasn't the issue; feature selection was. By shifting focus from "Logins" to "Feature Depth," we saved the company approximately **$200k ARR** per quarter.
