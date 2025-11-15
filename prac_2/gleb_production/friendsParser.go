package main

import (
	"encoding/json"
	"os"
)

type FullItem struct {
	ID              int    `json:"id"`
	Nickname        string `json:"nickname"`
	TrackCode       string `json:"track_code"`
	FirstName       string `json:"first_name"`
	LastName        string `json:"last_name"`
	CanAccessClosed bool   `json:"can_access_closed"`
	IsClosed        bool   `json:"is_closed"`
}

type LiteItem struct {
	ID        int    `json:"id"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
}

type Response struct {
	Items []FullItem `json:"items"`
}

type Root struct {
	Response Response `json:"response"`
}

type LiteResponse struct {
	Response struct {
		Items []LiteItem `json:"friends"`
	} `json:"response"`
}

func main() {
	file, err := os.Open("beforeParse.json")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	var fullData Root
	err = json.NewDecoder(file).Decode(&fullData)
	if err != nil {
		panic(err)
	}

	liteData := LiteResponse{
		Response: struct {
			Items []LiteItem `json:"friends"`
		}{
			Items: make([]LiteItem, len(fullData.Response.Items)),
		},
	}

	for i, item := range fullData.Response.Items {
		liteData.Response.Items[i] = LiteItem{
			ID:        item.ID,
			FirstName: item.FirstName,
			LastName:  item.LastName,
		}
	}

	outputFile, err := os.Create("parsed.json") // тот же файл
	if err != nil {
		panic(err)
	}
	defer outputFile.Close()

	err = json.NewEncoder(outputFile).Encode(liteData.Response.Items)
	if err != nil {
		panic(err)
	}
}
