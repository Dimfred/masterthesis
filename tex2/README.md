# README

## Introduction

This is the LME thesis template. It supports English and German, different types of theses as well as pdflatex and lualatex. For this reason, some configuration has to be done in `thesis_main.tex` before you can start.

## Thesis type

You can use this template for Bachelor's and Master's theses as well as dissertations. You need to set a parameter to change the thesis type. Look for the %CONFIG tag in `thesis_main.tex`. These are the places you need to change. For dissertations, you also need to use a different kind of title page, so please fill out the correct one.

## Language

By default, the template creates an English document. If you want/need to write a German document, check the %LANGUAGE tags in the main tex file - these are the places you need to change. The legal note ("Ich versichere...") has to be in German.

## Engine

The template supports pdfLaTeX and luaLaTeX, if the later is preferred the TeX Live version must be 2020 (or above). Both engines may produce a slightly different output (e.g. with a slightly different font).
